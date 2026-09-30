"""
Train the cross-episode task-memory branch (vla/task_memory.py) on cached tokens.

Inputs are the token cache (cache_tokens.py) and the demo experience library (build_experiences.py). The base
MemoryVLA stays frozen; only the branch is trained, with the original diffusion loss through the frozen DiT:
  - each step samples --episodes_per_step training episodes and --group_size sorted frames from each (as the original
    `group` dataloader) and replays the frozen episodic memory over them;
  - each episode gets --k experiences of its task, recorded in other episodes and other scenes (first-frame similarity
    below --same_scene_threshold); with probability --p_distractor they come from another task instead;
  - --exp_content noise trains the control model: everything identical, but experiences are Gaussian noise with the
    task's per-dimension statistics.
Every --val_every steps, held-out episodes are scored with real / noise / other-task / no experiences on identical
diffusion draws, so the conditions are compared pairwise.

    python script/train/task_memory/train_task_memory.py --ckpt /path/to/memvla-mikasa.pt \
        --cache_dir ./cache/task_memory/tokens --library ./cache/task_memory/demo_library.pt \
        --out_dir ./log/task_memory/main --exp_content real
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
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class GroupStream(IterableDataset):
    """Endless stream of (library index, `group_size` sorted frames) drawn from the given episodes."""

    def __init__(self, entries, cache_dir, group_size, seed):
        self.entries, self.cache_dir, self.group_size, self.seed = entries, Path(cache_dir), group_size, seed

    def __iter__(self):
        worker = get_worker_info()
        rng = np.random.default_rng(self.seed + (worker.id if worker else 0) * 7919)
        while True:
            entry = self.entries[rng.integers(len(self.entries))]
            yield load_group(entry, self.cache_dir, self.group_size, rng)


def load_group(entry, cache_dir, group_size, rng):
    data = torch.load(cache_dir / f"{entry['file']}.pt", map_location="cpu", weights_only=False)
    per = np.load(cache_dir / f"{entry['file']}.per.npy", mmap_mode="r")
    T = data["cog"].shape[0]
    if T >= group_size:
        frames = np.sort(rng.choice(T, group_size, replace=False))
    else:  # like the original group sampler: pad with the last frame
        frames = np.concatenate([np.arange(T), np.full(group_size - T, T - 1)])
    return {
        "lib_index": entry["lib_index"],
        "cog": data["cog"][frames],
        "per": torch.from_numpy(np.ascontiguousarray(per[frames])),
        "actions": data["actions"][frames],
        "timesteps": data["timesteps"][frames],
    }


class ExperiencePicker:
    """Chooses experiences from the library for training episodes; see the module docstring for the rules."""

    def __init__(self, library, entries, args, device):
        self.cog, self.per, self.meta = library["cog"], library["per"], library["meta"]
        self.noise_stats = {k[len("key/"):]: v for k, v in library["noise_stats"].items() if k.startswith("key/")}
        self.k, self.p_distractor, self.content, self.device = args.k, args.p_distractor, args.exp_content, device

        by_task = collections.defaultdict(list)
        for i, meta in enumerate(self.meta):
            by_task[meta["task_key"]].append(i)
        self.task_of = {i: meta["task_key"] for i, meta in enumerate(self.meta)}
        self.other_task = {task: [i for t, idx in by_task.items() if t != task for i in idx] for task in by_task}
        self.has_other_tasks = len(by_task) > 1

        # Candidates per episode: same task, another episode, and not recorded in the same scene
        self.allowed, dropped = {}, []
        wanted = {entry["lib_index"] for entry in entries}
        for task, indices in by_task.items():
            signature = F.normalize(self.per[indices, 0].flatten(1).to(device, torch.float16).float(), dim=-1)
            sims = (signature @ signature.T).cpu()
            for row, i in enumerate(indices):
                if i not in wanted:
                    continue
                keep = (sims[row] < args.same_scene_threshold).tolist()
                self.allowed[i] = [j for j, ok in zip(indices, keep) if ok and j != i]
                dropped.append(len(indices) - 1 - len(self.allowed[i]))
        print(f"[picker] same-scene exclusions per episode: mean={np.mean(dropped):.2f} max={max(dropped)}")

    def noise(self, task, generator):
        stats = self.noise_stats[task]
        K, N = self.per.shape[1], self.per.shape[2]
        shape_cog, shape_per = (self.k, K, self.cog.shape[-1]), (self.k, K, N, self.per.shape[-1])
        cog = stats["cog_mean"].to(self.device) + torch.randn(shape_cog, generator=generator, device=self.device) * \
            stats["cog_std"].to(self.device)
        per = stats["per_mean"].to(self.device) + torch.randn(shape_per, generator=generator, device=self.device) * \
            stats["per_std"].to(self.device)
        return cog, per

    def indices(self, lib_index, condition, rng):
        pool = self.other_task[self.task_of[lib_index]] if condition == "other_task" else self.allowed[lib_index]
        return rng.choice(pool, self.k, replace=len(pool) < self.k).tolist()

    def pick(self, lib_indices, condition, rng, generator):
        """[B, k, K, D_cog] and [B, k, K, N, D_per] experiences for a batch, on `device` (bf16)."""
        cogs, pers = [], []
        for lib_index in lib_indices:
            if condition == "noise":
                cog, per = self.noise(self.task_of[lib_index], generator)
            else:
                chosen = self.indices(lib_index, condition, rng)
                cog, per = self.cog[chosen].to(self.device), self.per[chosen].to(self.device)
            cogs.append(cog)
            pers.append(per)
        return torch.stack(cogs).to(torch.bfloat16), torch.stack(pers).to(torch.bfloat16)

    def conditions(self):
        """Validation conditions besides `none`."""
        return ("real", "noise", "other_task") if self.has_other_tasks else ("real", "noise")

    def training_condition(self, rng):
        if self.content == "noise":
            return "noise"
        return "other_task" if self.has_other_tasks and rng.random() < self.p_distractor else "real"


def forward_loss(parts, task_memory, batch, cog_exp, per_exp, repeated_steps, noise=None, t=None):
    """Per-sample diffusion loss for a batch of groups; `cog_exp is None` scores the base model (no task memory)."""
    from vla.task_memory import diffusion_loss, replay_episodic_memory

    cog, per = batch["cog"], batch["per"]  # [E, G, D], [E, G, N, Dp] bf16
    E, G = cog.shape[:2]
    fused_cog, fused_per = replay_episodic_memory(parts.cog_mem_bank, parts.per_mem_bank, cog, per, batch["timesteps"])
    cog_in, per_in = fused_cog.reshape(E * G, 1, -1), fused_per.reshape(E * G, *per.shape[2:])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if cog_exp is not None:
            d_cog, d_per = task_memory(
                cog.reshape(E * G, 1, -1), per.reshape(E * G, *per.shape[2:]),
                cog_exp.repeat_interleave(G, dim=0), per_exp.repeat_interleave(G, dim=0),
            )
            cog_in, per_in = cog_in + d_cog, per_in + d_per
        actions = batch["actions"].reshape(E * G, *batch["actions"].shape[2:])
        return diffusion_loss(
            parts.action_model, actions.repeat(repeated_steps, 1, 1), cog_in.repeat(repeated_steps, 1, 1),
            per_in.repeat(repeated_steps, 1, 1), noise=noise, t=t,
        )


def to_device(batch, device):
    return {
        "lib_index": [int(i) for i in batch["lib_index"]],
        "cog": batch["cog"].to(device, torch.bfloat16, non_blocking=True),
        "per": batch["per"].to(device, torch.bfloat16, non_blocking=True),
        "actions": batch["actions"].to(device, torch.float32, non_blocking=True),
        "timesteps": batch["timesteps"].to(device, non_blocking=True),
    }


def build_validation(val_entries, args, picker, parts, device):
    """Fixed held-out groups with fixed experiences and diffusion draws, reused at every evaluation."""
    rng = np.random.default_rng(args.seed + 1)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    chosen = [val_entries[i] for i in rng.permutation(len(val_entries))[: args.val_groups]]
    batches = []
    for start in range(0, len(chosen), args.episodes_per_step):
        groups = [load_group(e, Path(args.cache_dir), args.group_size, rng) for e in chosen[start : start + args.episodes_per_step]]
        batch = to_device({k: (torch.stack([g[k] for g in groups]) if k != "lib_index" else [g[k] for g in groups])
                           for k in groups[0]}, device)
        n = len(groups) * args.group_size * args.repeated_diffusion_steps
        batch["noise"] = torch.randn((n, *batch["actions"].shape[2:]), generator=generator, device=device)
        batch["t"] = torch.randint(0, parts.action_model.diffusion.num_timesteps, (n,), generator=generator, device=device)
        batch["exp"] = {
            condition: picker.pick(batch["lib_index"], condition, rng, generator) for condition in picker.conditions()
        }
        batches.append(batch)
    return batches


@torch.no_grad()
def validate(parts, task_memory, batches, args):
    task_memory.eval()
    parts.action_model.eval()  # no condition dropout: identical draws across conditions
    losses = collections.defaultdict(list)
    for batch in batches:
        for condition in ("none", *batch["exp"]):
            cog_exp, per_exp = batch["exp"][condition] if condition != "none" else (None, None)
            loss = forward_loss(parts, task_memory, batch, cog_exp, per_exp, args.repeated_diffusion_steps,
                                noise=batch["noise"], t=batch["t"])
            # One value per group: frames and diffusion repeats of a group are averaged
            E = len(batch["lib_index"])
            losses[condition].append(loss.view(args.repeated_diffusion_steps, E, args.group_size).mean(dim=(0, 2)).cpu())
    task_memory.train()
    parts.action_model.train()
    per_group = {c: torch.cat(v).double() for c, v in losses.items()}
    result = {f"loss_{c}": float(v.mean()) for c, v in per_group.items()}
    for other in [c for c in ("none", "noise", "other_task") if c in per_group]:
        diff = per_group["real"] - per_group[other]
        result[f"real_minus_{other}"] = float(diff.mean())
        result[f"real_minus_{other}_se"] = float(diff.std() / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--library", required=True, help="Demo library from build_experiences.py (--cache_dir)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--exp_content", choices=("real", "noise"), default="real", help="`noise` = control model")
    parser.add_argument("--k", type=int, default=1, help="Experiences per episode")
    parser.add_argument("--p_distractor", type=float, default=0.2)
    parser.add_argument("--same_scene_threshold", type=float, default=0.995)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--episodes_per_step", type=int, default=4)
    parser.add_argument("--repeated_diffusion_steps", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--cog_attn_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--val_groups", type=int, default=64)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_unknown_tasks", action="store_true", help="Drop episodes whose task id is -1")
    args = parser.parse_args()

    from vla.task_memory import TaskMemory, load_frozen_parts, save_task_memory

    torch.manual_seed(args.seed)
    device = "cuda"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    library = torch.load(args.library, map_location="cpu", weights_only=False)
    entries = []
    for i, meta in enumerate(library["meta"]):
        if meta["source"] != "demo" or (args.skip_unknown_tasks and meta["task_id"] == -1):
            continue
        entries.append({"lib_index": i, "file": meta["file"], "task_key": meta["task_key"]})
    picker = ExperiencePicker(library, entries, args, device)
    usable = [e for e in entries if picker.allowed.get(e["lib_index"])]
    if len(usable) < len(entries):
        print(f"dropping {len(entries) - len(usable)} episodes without a same-task experience from another scene")
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(usable))
    num_val = max(1, int(len(usable) * args.val_frac))
    val_entries = [usable[i] for i in permutation[:num_val]]
    train_entries = [usable[i] for i in permutation[num_val:]]
    print(f"episodes: train={len(train_entries)} val={len(val_entries)} "
          f"tasks={collections.Counter(e['task_key'] for e in usable)}")

    parts = load_frozen_parts(args.ckpt, device=device, dtype=torch.bfloat16)
    parts.action_model.train()  # condition dropout as in the original training; the weights stay frozen

    num_keyframes, cfg = library["cog"].shape[1], parts.config
    task_memory = TaskMemory(
        cog_dim=cfg["token_size"], per_dim=cfg["per_token_size"], cog_attn_dim=args.cog_attn_dim,
        per_attn_dim=cfg["per_token_size"], num_layers=args.num_layers, num_heads=args.num_heads,
        max_keyframes=max(32, num_keyframes), max_slots=max(8, args.k),
    ).to(device)
    print(f"task memory parameters: {sum(p.numel() for p in task_memory.parameters()) / 1e6:.1f}M")
    optimizer = torch.optim.AdamW(task_memory.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    val_batches = build_validation(val_entries, args, picker, parts, device)
    loader = DataLoader(
        GroupStream(train_entries, args.cache_dir, args.group_size, args.seed), batch_size=args.episodes_per_step,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    log = open(out_dir / "train_log.jsonl", "a")

    def record(row):
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    val = validate(parts, task_memory, val_batches, args)
    record({"step": 0, **val})  # zero-initialized output: every condition must equal `none` here
    generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    running, started = collections.defaultdict(float), time.time()
    for step, raw in zip(range(1, args.steps + 1), loader):
        batch = to_device(raw, device)
        condition = picker.training_condition(rng)
        cog_exp, per_exp = picker.pick(batch["lib_index"], condition, rng, generator)
        diag = step % args.log_every == 0
        task_memory.set_collect_diag(diag)
        loss = forward_loss(parts, task_memory, batch, cog_exp, per_exp, args.repeated_diffusion_steps).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(task_memory.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        running["loss"] += float(loss)
        running[f"n_{condition}"] += 1
        if diag:
            record({
                "step": step, "loss": running["loss"] / args.log_every, "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm), "steps_per_s": step / (time.time() - started),
                **{k: v for k, v in running.items() if k.startswith("n_")}, **task_memory.last_diagnostics,
            })
            running.clear()
        if step % args.val_every == 0 or step == args.steps:
            val = validate(parts, task_memory, val_batches, args)
            record({"step": step, **val})
        if step % args.save_every == 0 or step == args.steps:
            extra = {"step": step, "args": vars(args), "val": val, "exp_content": args.exp_content}
            save_task_memory(task_memory, out_dir / f"task_memory_step{step:06d}.pt", **extra)
            save_task_memory(task_memory, out_dir / "task_memory_last.pt", **extra)
    log.close()


if __name__ == "__main__":
    main()
