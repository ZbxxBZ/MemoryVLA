"""Summarize per-step attention, gate, and shadow-action diagnostics."""

import argparse
import glob
import json
import math
import statistics
from collections import defaultdict


def describe(values):
    if not values:
        return "n=0"
    values = sorted(values)
    mean = statistics.mean(values)
    median = statistics.median(values)
    p10 = values[max(0, math.floor(0.1 * (len(values) - 1)))]
    p90 = values[min(len(values) - 1, math.ceil(0.9 * (len(values) - 1)))]
    return f"n={len(values)} mean={mean:.4f} median={median:.4f} p10={p10:.4f} p90={p90:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag_dir", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for path in glob.glob(f"{args.diag_dir}/diag_steps*.jsonl")
            for line in open(path, encoding="utf-8") if line.strip()]
    print(f"Loaded {len(rows)} diagnostic steps from {args.diag_dir}")
    by_mode = defaultdict(list)
    for row in rows:
        by_mode[(row.get("task_id"), row.get("exp_mode"))].append(row)
    for (task, mode), group in sorted(by_mode.items(), key=lambda x: (x[0][0] or -1, x[0][1] or "")):
        print(f"\nTask={task} mode={mode} episodes="
              f"{len({(r.get('task_id'),r.get('init_id'),r.get('trial')) for r in group})}")
        for role in ("cog", "per"):
            attention = [r[f"{role}_attention_share"] for r in group if r.get(f"{role}_attention_share") is not None]
            uniform = [r[f"{role}_uniform_share"] for r in group if r.get(f"{role}_uniform_share") is not None]
            ratios = [a / u for a, u in zip(attention, uniform) if u > 0]
            gates = [r[f"{role}_gate_scale"] for r in group if r.get(f"{role}_gate_scale") is not None]
            print(f"  {role} attention: {describe(attention)}")
            print(f"  {role} uniform reference: {describe(uniform)}")
            print(f"  {role} attention/reference ratio: {describe(ratios)}")
            print(f"  {role} gate scale: {describe(gates)}")
        offsets = [r["action_offset_l2"] for r in group if r.get("action_offset_l2") is not None]
        first_by_episode = {}
        for row in group:
            key = (row.get("task_id"), row.get("init_id"), row.get("trial"))
            first_by_episode[key] = min(first_by_episode.get(key, row["step"]), row["step"])
        first_offsets = [r["action_offset_l2"] for r in group
                         if r.get("action_offset_l2") is not None and
                         r["step"] == first_by_episode[(r.get("task_id"),r.get("init_id"),r.get("trial"))]]
        print(f"  action offset first step: {describe(first_offsets)}")
        print(f"  action offset episode mean values: {describe(offsets)}")


if __name__ == "__main__":
    main()
