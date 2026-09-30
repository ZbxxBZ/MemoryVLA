"""
Analyze the MIKASA-Robo evaluation of the trained task memory (probe_task_memory.sh).

Groups are (model_tag, exp_mode, seed_offset). Every condition runs the same episodes (scene and diffusion seed), so
contrasts use the CMH test stratified by (task, init), Holm-corrected over the primary family, plus a matched risk
difference with a 95% CI from a bootstrap over (task, init) strata and a paired McNemar test where episodes pair up.
The "base" group is the task model with no experience (its branch is idle), pooled over seeds 1000 and 2000.
"""

import argparse
import collections
import glob
import json
import os
import random

import numpy as np

from analyze_round2 import TASKS, cmh, holm, mcnemar

BASE = ("task", "none", (1000, 2000))
CONTRASTS = [
    ("primary", "task real vs control real", ("task", "other_init_success", (1000,)), ("control", "other_init_success", (1000,))),
    ("primary", "task real vs task noise", ("task", "other_init_success", (1000,)), ("task", "noise", (1000,))),
    ("secondary", "task real vs base", ("task", "other_init_success", (1000,)), BASE),
    ("secondary", "control real vs base", ("control", "other_init_success", (1000,)), BASE),
    ("secondary", "task noise vs base", ("task", "noise", (1000,)), BASE),
    ("secondary", "task other-task vs base", ("task", "other_task_success", (1000,)), BASE),
    ("colour", "same vs different colour (RC)", ("task_rollout", "other_init_success_same_color", (1000,)),
     ("task_rollout", "other_init_success_diff_color", (1000,))),
]


def load_rows(log_dir):
    latest = {}
    for path in glob.glob(os.path.join(log_dir, "*_results.jsonl")):
        mtime = os.path.getmtime(path)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row.setdefault("model_tag", "base")
                key = (row["model_tag"], row["task_id"], row["init_id"], row["trial"], row["exp_mode"],
                       row["seed_offset"], bool(row["record"]))
                if key not in latest or mtime >= latest[key][0]:
                    latest[key] = (mtime, row)
    return [row for _, row in latest.values() if not row["record"]]


def usable(row):
    return row["exp_mode"] == "none" or (row.get("num_pinned_cog", 0) > 0 and row.get("num_pinned_per", 0) > 0)


def select(rows, spec, task_id=None):
    tag, mode, offsets = spec
    return [r for r in rows if r["model_tag"] == tag and r["exp_mode"] == mode and r["seed_offset"] in offsets
            and (task_id is None or r["task_id"] == task_id)]


def strata(rows):
    out = collections.defaultdict(list)
    for r in rows:
        out[(r["task_id"], r["init_id"])].append(int(bool(r["success"])))
    return out


def matched_diff(a_rows, b_rows, iters=4000, seed=0):
    """Weighted risk difference over (task, init) strata present in both groups, with a stratum-bootstrap 95% CI."""
    a, b = strata(a_rows), strata(b_rows)
    keys = sorted(set(a) & set(b))
    if not keys:
        return None

    def estimate(sample):
        num = den = 0.0
        for key in sample:
            va, vb = a[key], b[key]
            w = len(va) * len(vb) / (len(va) + len(vb))
            num += w * (sum(va) / len(va) - sum(vb) / len(vb))
            den += w
        return num / den

    rng = random.Random(seed)
    boots = sorted(estimate([rng.choice(keys) for _ in keys]) for _ in range(iters))
    return estimate(keys), boots[int(0.025 * iters)], boots[int(0.975 * iters)], len(keys)


def paired(a_rows, b_rows):
    a = {(r["task_id"], r["init_id"], r["trial"]): bool(r["success"]) for r in a_rows}
    b = {(r["task_id"], r["init_id"], r["trial"]): bool(r["success"]) for r in b_rows}
    return mcnemar(a, b)


def rate(rows):
    return f"{sum(bool(r['success']) for r in rows)}/{len(rows)} ({np.mean([bool(r['success']) for r in rows]):.1%})" \
        if rows else "0/0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--round2_dir", default=None, help="Round-2 results, to check the baseline reproduces")
    args = parser.parse_args()
    rows = load_rows(args.log_dir)
    excluded = collections.Counter((r["model_tag"], r["exp_mode"]) for r in rows if not usable(r))
    rows = [r for r in rows if usable(r)]
    print(f"Rows: {len(rows)} usable; excluded without experiences: {dict(excluded) or 'none'}")

    print("\n== Success by group ==")
    groups = sorted({(r["model_tag"], r["exp_mode"], r["seed_offset"]) for r in rows})
    print(f"{'model':<14}{'mode':<32}{'seed':>6}  {'all':>18}  " + "  ".join(f"{TASKS.get(t, t):>16}" for t in TASKS))
    for tag, mode, offset in groups:
        group = select(rows, (tag, mode, (offset,)))
        per_task = "  ".join(f"{rate([r for r in group if r['task_id'] == t]):>16}" for t in TASKS)
        print(f"{tag:<14}{mode:<32}{offset:>6}  {rate(group):>18}  {per_task}")

    if args.round2_dir:
        old = {(r["task_id"], r["init_id"], r["trial"]): bool(r["success"]) for r in load_rows(args.round2_dir)
               if r["exp_mode"] == "none" and r["seed_offset"] == 1000}
        new = {(r["task_id"], r["init_id"], r["trial"]): bool(r["success"])
               for r in select(rows, ("task", "none", (1000,)))}
        common = set(old) & set(new)
        same = sum(old[key] == new[key] for key in common)
        print(f"\nBaseline reproducibility vs round 2 (none@1000): {same}/{len(common)} episodes identical")

    print("\n== Contrasts (CMH by task/init; matched difference with stratum-bootstrap 95% CI) ==")
    primary = []
    for family, name, spec_a, spec_b in CONTRASTS:
        a, b = select(rows, spec_a), select(rows, spec_b)
        if not a or not b:
            print(f"[{family}] {name}: missing data")
            continue
        obs, expected, z, p = cmh(a, b, lambda r: (r["task_id"], r["init_id"]))
        diff = matched_diff(a, b)
        line = f"[{family}] {name}: {rate(a)} vs {rate(b)}; CMH z={z:.2f} p={p:.4f}"
        if diff:
            line += f"; matched diff={diff[0]:+.1%} [{diff[1]:+.1%}, {diff[2]:+.1%}] over {diff[3]} strata"
        if spec_a[2] == spec_b[2] == (1000,):
            n, b_count, c_count, _, p_pair = paired(a, b)
            line += f"; paired n={n} better={b_count} worse={c_count} McNemar p={p_pair:.4f}"
        print(line)
        if family == "primary":
            primary.append((name, p))
        for task_id, task in TASKS.items():
            ta, tb = select(rows, spec_a, task_id), select(rows, spec_b, task_id)
            if ta and tb:
                d = matched_diff(ta, tb)
                print(f"    {task:<4} {rate(ta)} vs {rate(tb)}" + (f"; diff={d[0]:+.1%} [{d[1]:+.1%}, {d[2]:+.1%}]" if d else ""))

    if primary:
        print("\nHolm-adjusted primary contrasts:")
        for (name, p), adjusted in zip(primary, holm([p for _, p in primary])):
            print(f"  {name:<32} raw_p={p:.4f} holm_p={adjusted:.4f}")

    sims = collections.defaultdict(list)
    for r in rows:
        if r.get("max_scene_sim") is not None:
            sims[(r["model_tag"], r["exp_mode"])].append(r["max_scene_sim"])
    if sims:
        print("\n== Closest library scene per episode (first-frame similarity; the threshold excludes near-identical) ==")
        for key, values in sorted(sims.items()):
            values = np.asarray(values)
            print(f"  {key[0]:<14}{key[1]:<32} median={np.median(values):.4f} max={values.max():.4f}")


if __name__ == "__main__":
    main()
