"""
Summarize the experience-prefill probe (see probe_experience.sh).

Rows with `record=True` are phase 1 (collection); they define each initial state's difficulty:
    always = succeeded in every collection rollout, never = failed in all of them, mixed = otherwise.
Rows with `record=False` are phase 2; each mode is paired against `none` on (task, init, trial, seed_offset),
i.e. the same initial state and the same diffusion seed, and tested with an exact McNemar test.
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

MODE_ORDER = ("none", "other_init_success", "same_init_success", "same_init_failure")
CATEGORIES = ("all", "mixed", "never", "always")


def load_rows(log_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(log_dir, "**", "*_results.jsonl"), recursive=True)):
        with open(path, "r") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def mcnemar_p(b, c):
    """Exact two-sided McNemar test on the discordant pair counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    args = parser.parse_args()

    rows = load_rows(args.log_dir)
    collect = [r for r in rows if r["record"]]
    evals = [r for r in rows if not r["record"]]
    print(f"Loaded {len(collect)} collection rows and {len(evals)} evaluation rows from `{args.log_dir}`\n")

    # Phase 1: difficulty of each initial state
    outcomes = defaultdict(list)
    for r in collect:
        outcomes[(r["suite"], r["task_id"], r["init_id"])].append(r["success"])

    def category(state):
        s = outcomes.get(state)
        if not s:
            return "unknown"
        rate = sum(s) / len(s)
        return "always" if rate == 1 else ("never" if rate == 0 else "mixed")

    print("== Phase 1: collection ==")
    print(f"{'task':>6} {'inits':>6} {'success':>8} {'always':>7} {'mixed':>6} {'never':>6}")
    for task in sorted({state[:2] for state in outcomes}):
        states = [s for s in outcomes if s[:2] == task]
        runs = [x for s in states for x in outcomes[s]]
        cats = [category(s) for s in states]
        print(
            f"{task[1]:>6} {len(states):>6} {sum(runs) / len(runs):>8.1%} "
            f"{cats.count('always'):>7} {cats.count('mixed'):>6} {cats.count('never'):>6}"
        )

    # Phase 2: success per mode and difficulty category, paired against `none`
    by_mode = defaultdict(dict)
    for r in evals:
        key = (r["suite"], r["task_id"], r["init_id"], r["trial"], r["seed_offset"])
        by_mode[r["exp_mode"]][key] = r["success"]
    modes = [m for m in MODE_ORDER if m in by_mode] + sorted(m for m in by_mode if m not in MODE_ORDER)

    def in_category(key, cat):
        return cat == "all" or category(key[:3]) == cat

    print("\n== Phase 2: success rate (n) ==")
    print(f"{'mode':<20}" + "".join(f"{c:>16}" for c in CATEGORIES))
    for mode in modes:
        cells = []
        for cat in CATEGORIES:
            vals = [v for k, v in by_mode[mode].items() if in_category(k, cat)]
            cells.append(f"{sum(vals) / len(vals):.1%} ({len(vals)})" if vals else "-")
        print(f"{mode:<20}" + "".join(f"{c:>16}" for c in cells))

    if "none" not in by_mode:
        print("\nNo `none` runs found; skipping paired comparison.")
        return

    print("\n== Phase 2: paired vs `none` (delta = success change; b = fixed, c = broken) ==")
    print(f"{'mode':<20} {'cat':<7} {'pairs':>6} {'delta':>8} {'b':>4} {'c':>4} {'p':>8}")
    for mode in modes:
        if mode == "none":
            continue
        for cat in CATEGORIES:
            keys = [k for k in by_mode[mode] if k in by_mode["none"] and in_category(k, cat)]
            if not keys:
                continue
            b = sum(1 for k in keys if by_mode[mode][k] and not by_mode["none"][k])
            c = sum(1 for k in keys if not by_mode[mode][k] and by_mode["none"][k])
            print(
                f"{mode:<20} {cat:<7} {len(keys):>6} {(b - c) / len(keys):>+8.1%} "
                f"{b:>4} {c:>4} {mcnemar_p(b, c):>8.3f}"
            )


if __name__ == "__main__":
    main()
