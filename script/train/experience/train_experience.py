"""
Train the experience adapter (vla/experience_retrieval.py) on cached tokens; the base MemoryVLA stays frozen and only
the adapter learns, with the original diffusion loss through the frozen DiT.

Every step:
  - samples --episodes_per_step training demos and --group_size calls of each;
  - gives each demo m ~ U{1..--num_refs} references: other demos of the same task (never itself, nor a near-identical
    scene), at random or, with probability --p_similar, among the 2m most similar first frames (deploy.py selects the
    most similar); with probability --p_none it gets none (the base policy) and with --p_distractor demos of another
    task;
  - aligns every reference to the demo with the forward-only pointer (pointer_align), as the server does call by call;
  - scores the calls; the episodic memory is replayed over each whole demo beforehand, as deployment runs it.
--ref_content trains control adapters: `unaligned` reads a random step of each reference instead of the aligned one.
Every --val_every steps, held-out demos are scored with aligned / unaligned / other-task / no references on identical
diffusion draws, so the conditions are compared pairwise.

    PYTHONPATH=. python script/train/experience/train_experience.py --ckpt /path/to/run/checkpoints/x.pt \
        --cache_dir ./cache/experience/goal_tokens --out_dir ./log/experience/goal_main
"""

import argparse
import collections
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from exp_common import (
    diffusion_loss, load_cache_episode, load_cache_index, load_frozen_parts, replay_episodic_memory, scene_signature,
    task_key_of, unit_keys,
)
from vla.experience_retrieval import SCALAR_NAMES, ExperienceAdapter, pointer_align, reference_scalars, save_adapter


class Demos:
    """Cached demos in host memory: keys, action chunks, first-frame signatures and replayed fused tokens."""

    def __init__(self, cache_dir, parts, max_episodes=0):
        entries = load_cache_index(cache_dir)
        if max_episodes:
            entries = entries[:max_episodes]
        strides = collections.Counter(e.get("frame_stride", 1) for e in entries)
        print(f"[demos] {len(entries)} episodes, frame strides {dict(strides)}", flush=True)
        self.files, self.task, self.keys, self.actions, self.sig, self.fused_cog, self.fused_per = ([] for _ in range(7))
        started = time.time()
        for n, entry in enumerate(entries, 1):
            data = load_cache_episode(cache_dir, entry)
            cog = data["cog"].to("cuda", torch.bfloat16)
            per = data["per"].to("cuda", torch.bfloat16)
            fused_cog, fused_per = replay_episodic_memory(parts.cog_mem_bank, parts.per_mem_bank, cog, per)
            self.files.append(entry["file"])
            self.task.append(task_key_of(entry))
            self.keys.append(unit_keys(data["cog"]))
            self.actions.append(data["actions"])
            self.sig.append(scene_signature(data["per"][0]))
            self.fused_cog.append(fused_cog.half().cpu())
            self.fused_per.append(fused_per.half().cpu())
            if n % 100 == 0:
                print(f"[demos] replayed {n}/{len(entries)} ({time.time() - started:.0f}s)", flush=True)
        self.by_task = collections.defaultdict(list)
        for i, task in enumerate(self.task):
            self.by_task[task].append(i)
        print(f"[demos] tasks: { {t: len(v) for t, v in self.by_task.items()} }", flush=True)

    def __len__(self):
        return len(self.files)


class ReferencePicker:
    """Reference episodes for a demo (see the module docstring); the pools only hold training demos."""

    def __init__(self, demos: Demos, train_indices, args):
        self.demos, self.args = demos, args
        train = set(train_indices)
        sig = torch.stack(demos.sig).float().cuda()
        sims = (sig @ sig.T).cpu()
        self.same, self.other = {}, {}
        dropped = []
        for i in range(len(demos)):
            task = demos.task[i]
            pool = [j for j in demos.by_task[task] if j in train and j != i]
            keep = [j for j in pool if sims[i, j] < args.same_scene_threshold]
            dropped.append(len(pool) - len(keep))
            self.same[i] = sorted(keep, key=lambda j: -float(sims[i, j]))  # most similar first
            self.other[i] = [j for j in train if demos.task[j] != task]
        print(f"[picker] near-identical scenes dropped per demo: mean={np.mean(dropped):.2f} max={max(dropped)}")
        self.has_other_tasks = any(self.other[i] for i in range(len(demos)))

    def pick(self, i, condition, rng, num=None):
        """Reference indices for demo i under `condition` (real / unaligned / other_task / none)."""
        if condition == "none":
            return []
        m = num if num is not None else int(rng.integers(1, self.args.num_refs + 1))
        if condition == "other_task":
            pool = self.other[i]
            return [pool[k] for k in rng.permutation(len(pool))[:m]] if pool else []
        pool = self.same[i]
        if not pool:
            return []
        if rng.random() < self.args.p_similar:
            top = pool[: 2 * m]
            return [top[k] for k in rng.permutation(len(top))[:m]]
        return [pool[k] for k in rng.permutation(len(pool))[:m]]

    def training_condition(self, rng):
        u = rng.random()
        if u < self.args.p_none:
            return "none"
        if self.has_other_tasks and u < self.args.p_none + self.args.p_distractor:
            return "other_task"
        return self.args.ref_content


def reference_batch(demos, episode, frames, refs, unaligned, rng, max_refs, window):
    """Adapter inputs for `frames` of `episode` reading `refs`: dict of [G, max_refs, ...] tensors (CPU)."""
    G = len(frames)
    chunk = demos.actions[episode].shape[1:]
    actions = torch.zeros(G, max_refs, *chunk)
    scalars = torch.zeros(G, max_refs, len(SCALAR_NAMES))
    success = torch.zeros(G, max_refs, dtype=torch.long)
    valid = torch.zeros(G, max_refs, dtype=torch.bool)
    query = demos.keys[episode].float()
    for m, r in enumerate(refs[:max_refs]):
        sims = (query @ demos.keys[r].float().T).numpy()  # [T, T_r]
        T_r = sims.shape[1]
        js = pointer_align(sims, window)
        for g, f in enumerate(frames):
            j = int(rng.integers(T_r)) if unaligned else int(js[f])
            actions[g, m] = demos.actions[r][j]
            scalars[g, m] = torch.tensor(reference_scalars(j, float(sims[f, j]), int(f), T_r))
            success[g, m] = 1
            valid[g, m] = True
    return {"actions": actions, "scalars": scalars, "success": success, "valid": valid}


def sample_frames(T, group_size, rng):
    if T >= group_size:
        return np.sort(rng.choice(T, group_size, replace=False))
    return np.concatenate([np.arange(T), np.full(group_size - T, T - 1)])  # pad with the last call


def build_batch(demos, picker, episodes, conditions, rng, args, refs_override=None, frames_override=None):
    """Batch of len(episodes) * group_size calls; conditions[k] applies to episodes[k]."""
    cog, per, actions, exp_parts = [], [], [], []
    for k, (episode, condition) in enumerate(zip(episodes, conditions)):
        if frames_override is not None:
            frames = frames_override[k]
        else:
            frames = sample_frames(demos.actions[episode].shape[0], args.group_size, rng)
        if refs_override is not None:
            refs = refs_override[k]
        else:
            refs = picker.pick(episode, "real" if condition == "unaligned" else condition, rng)
        exp_parts.append(reference_batch(demos, episode, frames, refs, condition == "unaligned", rng,
                                         args.max_refs, args.window))
        cog.append(demos.fused_cog[episode][frames])
        per.append(demos.fused_per[episode][frames])
        actions.append(demos.actions[episode][frames])
    exp = {key: torch.cat([p[key] for p in exp_parts]) for key in exp_parts[0]}
    return {
        "cog": torch.cat(cog)[:, None].cuda().to(torch.bfloat16),  # [B, 1, D]
        "per": torch.cat(per).cuda().to(torch.bfloat16),  # [B, N, Dp]
        "actions": torch.cat(actions).cuda().float(),  # [B, F, A]
        "exp": {key: value.cuda() for key, value in exp.items()},
    }


def batch_loss(parts, batch, R, noise=None, t=None, use_exp=True):
    rep = lambda x: x.repeat(R, *([1] * (x.dim() - 1)))  # noqa: E731
    exp = {k: rep(v) for k, v in batch["exp"].items()} if use_exp else None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return diffusion_loss(parts.action_model, rep(batch["actions"]), rep(batch["cog"]), rep(batch["per"]),
                              exp=exp, noise=noise, t=t)


def build_validation(demos, picker, val_indices, args, num_timesteps):
    """
    Fixed held-out batches: identical calls and diffusion draws in every condition; aligned and unaligned read the
    same references, with --num_refs of them.
    """
    rng = np.random.default_rng(args.seed + 1)
    generator = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    chosen = [val_indices[i] for i in rng.permutation(len(val_indices))[: args.val_episodes]]
    batches = []
    for start in range(0, len(chosen), args.episodes_per_step):
        episodes = chosen[start : start + args.episodes_per_step]
        frames = [sample_frames(demos.actions[e].shape[0], args.group_size, rng) for e in episodes]
        refs = {c: [picker.pick(e, c, rng, num=args.num_refs) for e in episodes] for c in ("real", "other_task")}
        conditions = {}
        for condition in ("real", "unaligned", "other_task", "none"):
            if condition == "other_task" and not picker.has_other_tasks:
                continue
            source = refs["other_task"] if condition == "other_task" else refs["real"]
            conditions[condition] = build_batch(demos, picker, episodes, [condition] * len(episodes), rng, args,
                                                refs_override=source, frames_override=frames)
        n = len(episodes) * args.group_size * args.repeated_diffusion_steps
        first = next(iter(conditions.values()))
        noise = torch.randn((n, *first["actions"].shape[1:]), generator=generator, device="cuda")
        t = torch.randint(0, num_timesteps, (n,), generator=generator, device="cuda")
        batches.append({"conditions": conditions, "noise": noise, "t": t, "E": len(episodes)})
    return batches


@torch.no_grad()
def validate(parts, adapter, batches, args):
    adapter.eval()
    parts.action_model.eval()  # no condition dropout: identical draws across conditions
    losses = collections.defaultdict(list)
    for batch in batches:
        for condition, data in batch["conditions"].items():
            loss = batch_loss(parts, data, args.repeated_diffusion_steps, noise=batch["noise"], t=batch["t"],
                              use_exp=condition != "none")
            losses[condition].append(
                loss.view(args.repeated_diffusion_steps, batch["E"], args.group_size).mean(dim=(0, 2)).cpu())
    adapter.train()
    parts.action_model.train()
    per_episode = {c: torch.cat(v).double() for c, v in losses.items()}
    result = {f"val_{c}": float(v.mean()) for c, v in per_episode.items()}
    for other in ("none", "unaligned", "other_task"):
        if other in per_episode:
            diff = per_episode["real"] - per_episode[other]
            result[f"real_minus_{other}"] = float(diff.mean())
            result[f"real_minus_{other}_se"] = float(diff.std() / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="MemoryVLA checkpoint <run>/checkpoints/<name>.pt")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ref_content", choices=("real", "unaligned"), default="real",
                        help="`unaligned` = control adapter reading random reference steps")
    parser.add_argument("--num_refs", type=int, default=4, help="Maximum references per episode")
    parser.add_argument("--max_refs", type=int, default=8, help="Adapter slots (>= num_refs; deploy may use up to this)")
    parser.add_argument("--window", type=int, default=3, help="Pointer look-ahead in calls")
    parser.add_argument("--p_none", type=float, default=0.25)
    parser.add_argument("--p_distractor", type=float, default=0.1)
    parser.add_argument("--p_similar", type=float, default=0.5)
    parser.add_argument("--same_scene_threshold", type=float, default=0.999)
    parser.add_argument("--inject_every", type=int, default=2, help="Cross-attention after every n-th DiT block")
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--episodes_per_step", type=int, default=4)
    parser.add_argument("--repeated_diffusion_steps", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--val_episodes", type=int, default=64)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_episodes", type=int, default=0, help="0 = all (debugging aid)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    assert args.num_refs <= args.max_refs

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    parts = load_frozen_parts(args.ckpt)
    demos = Demos(args.cache_dir, parts, args.max_episodes)
    permutation = rng.permutation(len(demos))
    num_val = max(1, int(len(demos) * args.val_frac))
    val_indices, train_indices = permutation[:num_val].tolist(), permutation[num_val:].tolist()
    picker = ReferencePicker(demos, train_indices, args)
    print(f"episodes: train={len(train_indices)} val={len(val_indices)}", flush=True)

    cfg = parts.config
    adapter = ExperienceAdapter(
        hidden_size=cfg["hidden_size"], num_heads=cfg["num_heads"], depth=cfg["depth"],
        inject_every=args.inject_every, action_dim=cfg["action_dim"],
        chunk_len=cfg["future_action_window_size"] + 1, max_refs=args.max_refs,
    ).cuda()
    parts.action_model.net.attach_exp_adapter(adapter)  # fp32 trainable adapter inside the frozen bf16 DiT
    parts.action_model.train()  # condition dropout as in the original training; the base weights stay frozen
    params = [p for p in adapter.parameters() if p.requires_grad]
    print(f"adapter parameters: {sum(p.numel() for p in params) / 1e6:.1f}M, layers after blocks {adapter.inject_layers}")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    val_batches = build_validation(demos, picker, val_indices, args, parts.action_model.diffusion.num_timesteps)
    log = open(out_dir / "train_log.jsonl", "a")

    def record(row):
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    val = validate(parts, adapter, val_batches, args)
    record({"step": 0, **val})  # zero gates: every condition equals `none` here
    running, started = collections.defaultdict(float), time.time()
    for step in range(1, args.steps + 1):
        episodes = [train_indices[k] for k in rng.integers(len(train_indices), size=args.episodes_per_step)]
        conditions = [picker.training_condition(rng) for _ in episodes]
        batch = build_batch(demos, picker, episodes, conditions, rng, args)
        loss = batch_loss(parts, batch, args.repeated_diffusion_steps).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        running["loss"] += float(loss)
        for condition in conditions:
            running[f"n_{condition}"] += 1
        if step % args.log_every == 0:
            gates = adapter.gates()
            record({
                "step": step, "loss": running.pop("loss") / args.log_every, "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm), "steps_per_s": step / (time.time() - started),
                "gate_mean": float(np.mean(list(gates.values()))), "gate_max": float(np.max(np.abs(list(gates.values())))),
                **running,
            })
            running.clear()
        if step % args.val_every == 0 or step == args.steps:
            val = validate(parts, adapter, val_batches, args)
            record({"step": step, **val, **adapter.gates()})
        if step % args.save_every == 0 or step == args.steps:
            extra = {"step": step, "args": vars(args), "val": val, "ckpt": str(args.ckpt), "window": args.window}
            save_adapter(adapter, out_dir / f"adapter_step{step:06d}.pt", **extra)
            save_adapter(adapter, out_dir / "adapter_last.pt", **extra)
    log.close()


if __name__ == "__main__":
    main()
