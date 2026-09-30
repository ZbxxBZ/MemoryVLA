"""
Pack the experience library of the trained task-memory branch (read by vla.experience.TaskExperienceLibrary).

The frozen episodic memory of a MemoryVLA checkpoint is replayed over each recorded episode with deployment semantics
(one policy call every --call_stride environment steps, memory timestep +1 per call); the episodic *fused* tokens at
--num_keyframes uniform keyframes over the calls are kept, together with per-task statistics for the noise condition.

Sources:
    --cache_dir      token cache from cache_tokens.py (training demonstrations, all successful)
    --exp_store_dir  an ExperienceStore recorded by deploy.py, e.g. the round-2 phase-1 rollouts (init / colour known)

    python script/train/task_memory/build_experiences.py --ckpt /path/to/memvla-mikasa.pt \
        --cache_dir ./cache/task_memory/tokens --out ./cache/task_memory/demo_library.pt
"""

import argparse
import collections
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tm_common import infer_task, load_cache_index, load_task_map, quantiles


def demo_episodes(cache_dir, call_stride, suite, task_map):
    for entry in load_cache_index(cache_dir):
        task_key, task_id = infer_task(entry, task_map)
        data = torch.load(Path(cache_dir) / f"{entry['file']}.pt", map_location="cpu", weights_only=False)
        per = np.load(Path(cache_dir) / f"{entry['file']}.per.npy", mmap_mode="r")
        calls = list(range(0, entry["T"], call_stride))
        meta = {
            "name": f"demo_{entry['episode']}", "source": "demo", "episode": entry["episode"], "file": entry["file"],
            "suite": suite, "task_key": task_key, "task_id": task_id, "init_id": -1, "success": True,
            "target_color": None, "num_calls": len(calls),
        }
        yield meta, data["cog"][calls], torch.from_numpy(np.ascontiguousarray(per[calls])), torch.arange(len(calls))


def rollout_episodes(store_dir):
    from vla.experience import ExperienceStore

    store = ExperienceStore(store_dir)
    for meta in sorted(store.index.values(), key=lambda m: m["name"]):
        if meta.get("exp_mode", "none") != "none":
            continue  # only recordings made without prefilled experiences
        record = store.load(meta)
        out = {
            "name": f"rollout_{meta['name']}", "source": "rollout", "suite": meta["suite"],
            "task_key": f"task{int(meta['task_id'])}", "task_id": int(meta["task_id"]),
            "init_id": int(meta["init_id"]), "trial": int(meta.get("trial", -1)), "success": bool(meta["success"]),
            "target_color": meta.get("target_color"), "num_calls": int(record["cog"].shape[0]),
        }
        yield out, record["cog"], store.load_per(meta), torch.as_tensor(np.asarray(record["timesteps"]), dtype=torch.long)
        store._cache.pop(meta["name"], None)


def per_dim_stats(tensor: torch.Tensor, chunk: int = 64):
    """Mean / std over every leading dimension, accumulated in float64 chunks (the per library is several GB)."""
    total, total_sq, count = 0.0, 0.0, 0
    for start in range(0, tensor.shape[0], chunk):
        x = tensor[start : start + chunk].double().reshape(-1, tensor.shape[-1])
        total, total_sq, count = total + x.sum(0), total_sq + (x * x).sum(0), count + x.shape[0]
    mean = total / count
    return mean.float(), (total_sq / count - mean * mean).clamp_min(0).sqrt().float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--exp_store_dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--suite", default="mikasa")
    parser.add_argument("--num_keyframes", type=int, default=8)
    parser.add_argument("--call_stride", type=int, default=4, help="Env steps per policy call (action chunk length)")
    parser.add_argument("--task_map", default=None, help="JSON {substring: task_id}, see tm_common.infer_task")
    parser.add_argument("--max_per_task", type=int, default=0, help="Keep at most this many episodes per task (0 = all)")
    parser.add_argument("--batch_size", type=int, default=32, help="Episodes replayed together (same call count)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    assert (args.cache_dir is None) != (args.exp_store_dir is None), "Pass exactly one of --cache_dir / --exp_store_dir"

    from vla.task_memory import load_frozen_parts, replay_episodic_memory, stack_padded, uniform_keyframes

    parts = load_frozen_parts(args.ckpt, device="cuda", dtype=torch.bfloat16)
    if args.cache_dir:
        episodes = demo_episodes(args.cache_dir, args.call_stride, args.suite, load_task_map(args.task_map))
    else:
        episodes = rollout_episodes(args.exp_store_dir)

    metas, cogs, pers = [], [], []
    buckets = collections.defaultdict(list)  # episodes with equal call counts are replayed in one batch

    def flush(items):
        raw_cog = torch.stack([c for _, c, _, _ in items]).to("cuda", torch.bfloat16)
        raw_per = torch.stack([p for _, _, p, _ in items]).to("cuda", torch.bfloat16)
        steps = torch.stack([t for _, _, _, t in items]).to("cuda")
        fused_cog, fused_per = replay_episodic_memory(parts.cog_mem_bank, parts.per_mem_bank, raw_cog, raw_per, steps)
        keyframes = stack_padded(uniform_keyframes(raw_cog.shape[1], args.num_keyframes), args.num_keyframes)
        for b, (meta, _, _, _) in enumerate(items):
            metas.append(meta)
            cogs.append(fused_cog[b, keyframes].half().cpu())
            pers.append(fused_per[b, keyframes].half().cpu())

    for n, item in enumerate(episodes, 1):
        bucket = buckets[item[1].shape[0]]
        bucket.append(item)
        if len(bucket) >= args.batch_size:
            flush(bucket)
            bucket.clear()
        if n % 200 == 0:
            print(f"read {n} episodes, packed {len(metas)}", flush=True)
    for bucket in buckets.values():
        if bucket:
            flush(bucket)

    order = sorted(range(len(metas)), key=lambda i: metas[i]["name"])
    if args.max_per_task:
        rng, kept, by_task = np.random.RandomState(args.seed), [], collections.defaultdict(list)
        for i in order:
            by_task[metas[i]["task_key"]].append(i)
        for indices in by_task.values():
            kept += sorted(rng.permutation(indices)[: args.max_per_task].tolist())
        order = sorted(kept, key=lambda i: metas[i]["name"])
    metas = [metas[i] for i in order]
    cog, per = torch.stack([cogs[i] for i in order]), torch.stack([pers[i] for i in order])

    noise_stats, report = {}, []
    by_task = collections.defaultdict(list)
    for i, meta in enumerate(metas):
        by_task[(meta["suite"], meta["task_id"], meta["task_key"])].append(i)
    for (suite, task_id, task_key), indices in sorted(by_task.items(), key=lambda kv: str(kv[0])):
        cog_mean, cog_std = per_dim_stats(cog[indices])
        per_mean, per_std = per_dim_stats(per[indices])
        stats = {"cog_mean": cog_mean, "cog_std": cog_std, "per_mean": per_mean, "per_std": per_std}
        noise_stats[f"{suite}/{task_id}"] = stats
        noise_stats[f"key/{task_key}"] = stats

        # How close is each episode's first frame to its nearest other episode of the task? Same-scene recordings sit
        # near 1; the gap to other scenes sets --same_scene_threshold for training and deployment.
        signature = F.normalize(per[indices, 0].flatten(1).float().cuda(), dim=-1)
        sims = signature @ signature.T
        sims.fill_diagonal_(-1.0)
        nearest = sims.max(dim=1).values.cpu().numpy()
        successes = sum(metas[i]["success"] for i in indices)
        report.append(
            f"{suite} task_id={task_id:3d} key={task_key!r}: episodes={len(indices)} success={successes}\n"
            f"    nearest first-frame similarity: {quantiles(nearest)}\n"
            f"    share >= 0.99: {np.mean(nearest >= 0.99):.3f}, >= 0.995: {np.mean(nearest >= 0.995):.3f}, "
            f">= 0.999: {np.mean(nearest >= 0.999):.3f}"
        )

    config = {k: v for k, v in vars(args).items()}
    torch.save({"meta": metas, "cog": cog, "per": per, "noise_stats": noise_stats, "config": config}, args.out)
    print(f"Saved {len(metas)} experiences to {args.out}: cog {tuple(cog.shape)}, per {tuple(per.shape)}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
