"""Analyze Round 2 MIKASA experience-prefill results with stratified tests."""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

TASKS = {1: "IM", 2: "RC3", 3: "RC5"}
MODES = (
    "none", "other_init_success", "same_init_success", "same_init_failure",
    "other_task_success", "noise", "other_init_success_same_color", "other_init_success_diff_color",
)
MAIN_COMPARISONS = ("other_init_success", "same_init_success", "same_init_failure", "other_task_success", "noise")


def load_dedup(log_dir):
    latest = {}
    files = glob.glob(os.path.join(log_dir, "*_results.jsonl"))
    for path in files:
        mtime = os.path.getmtime(path)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["task_id"], row["init_id"], row["trial"], row["exp_mode"],
                       row["seed_offset"], bool(row["record"]))
                previous = latest.get(key)
                if previous is None or mtime >= previous[0]:
                    latest[key] = (mtime, row)
    return [item[1] for item in latest.values()], len(files)


def p_two_sided_z(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def cmh(group1, group0, stratum_fn):
    strata = defaultdict(lambda: [[], []])
    for row in group1:
        strata[stratum_fn(row)][0].append(bool(row["success"]))
    for row in group0:
        strata[stratum_fn(row)][1].append(bool(row["success"]))
    obs = expected = variance = 0.0
    for first, second in strata.values():
        n1, n0 = len(first), len(second)
        n = n1 + n0
        if not n1 or not n0 or n <= 1:
            continue
        s1, s0 = sum(first), sum(second)
        total = s1 + s0
        exp = n1 * total / n
        obs += s1
        expected += exp
        variance += n1 * n0 * total * (n - total) / (n * n * (n - 1))
    if variance <= 0:
        return obs, expected, 0.0, 1.0
    z = (obs - expected) / math.sqrt(variance)
    return obs, expected, z, p_two_sided_z(z)


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(order) - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def mcnemar(first, second):
    keys = set(first) & set(second)
    b = sum(first[k] and not second[k] for k in keys)
    c = sum(second[k] and not first[k] for k in keys)
    n = b + c
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2**n)) if n else 1.0
    return len(keys), b, c, (b - c) / len(keys) if keys else 0.0, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    args = parser.parse_args()
    rows, file_count = load_dedup(args.log_dir)
    collect = [r for r in rows if r["record"]]
    eval_rows = [r for r in rows if not r["record"]]
    exp_modes = [r for r in eval_rows if r["exp_mode"] != "none"]
    zero_pin = {mode: sum(r.get("num_pinned_cog", 0) == 0 or r.get("num_pinned_per", 0) == 0
                          for r in exp_modes if r["exp_mode"] == mode) for mode in MODES if mode != "none"}
    usable = [r for r in eval_rows if r["exp_mode"] == "none" or
              (r.get("num_pinned_cog", 0) > 0 and r.get("num_pinned_per", 0) > 0)]
    print(f"Read {file_count} top-level files; deduplicated rows={len(rows)} "
          f"(collect={len(collect)}, eval={len(eval_rows)}).")
    print("Rows excluded for zero pinned cog/per:", ", ".join(f"{m}={n}" for m, n in zero_pin.items()))

    # Per-init phase-1 success determines the predeclared difficulty categories.
    state_outcomes = defaultdict(list)
    for row in collect:
        state_outcomes[(row["task_id"], row["init_id"])].append(bool(row["success"]))

    def category(row):
        outcomes = state_outcomes.get((row["task_id"], row["init_id"]), [])
        if not outcomes:
            return "unknown"
        if all(outcomes):
            return "always"
        if not any(outcomes):
            return "never"
        return "mixed"

    by_mode = defaultdict(list)
    for row in usable:
        by_mode[(row["exp_mode"], row["seed_offset"])].append(row)
    baseline = by_mode[("none", 1000)] + by_mode[("none", 2000)]
    print(f"\nPhase 1 collect reference: {sum(r['success'] for r in collect)}/{len(collect)} "
          f"({sum(r['success'] for r in collect)/len(collect):.1%})")
    print("task  inits  collect  always mixed never")
    for task_id, task_name in TASKS.items():
        states = {state for state in state_outcomes if state[0] == task_id}
        task_rows = [r for r in collect if r["task_id"] == task_id]
        cats = [category({"task_id": task_id, "init_id": init}) for _, init in states]
        print(f"{task_name:<5} {len(states):>5}  {sum(r['success'] for r in task_rows)}/{len(task_rows):<3} "
              f"{cats.count('always'):>6} {cats.count('mixed'):>5} {cats.count('never'):>5}")

    pooled = defaultdict(list)
    for row in baseline:
        pooled[(row["task_id"], row["init_id"])].append(row)

    def get_mode(mode):
        return [r for r in usable if r["exp_mode"] == mode and r["seed_offset"] == 1000]

    def print_cmh_table(title, filter_fn=lambda r: True, strata_fn=lambda r: (r["task_id"], r["init_id"])):
        print(f"\n{title}")
        print(f"{'mode':<38} {'actual':>10} {'expected':>10} {'rate':>8} {'delta':>8} {'z':>8} {'p':>8}")
        output = {}
        for mode in MODES:
            if mode == "none":
                continue
            group = [r for r in get_mode(mode) if filter_fn(r)]
            control = [r for r in baseline if filter_fn(r)]
            if not group or not control:
                continue
            obs, expected, z, p = cmh(group, control, strata_fn)
            rate = sum(r["success"] for r in group) / len(group)
            # Compare against baseline rows from the same strata only: a mode that exists for a subset of
            # starts (e.g. same_init_success, or RC-only color modes) must not be compared with the whole baseline.
            group_strata = {strata_fn(r) for r in group}
            matched_control = [r for r in control if strata_fn(r) in group_strata] or control
            base_rate = sum(r["success"] for r in matched_control) / len(matched_control)
            print(f"{mode:<38} {int(obs):>4}/{len(group):<5} {expected:>10.2f} {rate:>7.1%} "
                  f"{rate-base_rate:>+7.1%} {z:>8.3f} {p:>8.4f}")
            output[mode] = p
        return output

    pvalues = print_cmh_table("== Main comparisons vs pooled none@1000 + none@2000 (CMH, task/init strata) ==")
    main_p = [(mode, pvalues[mode]) for mode in MAIN_COMPARISONS if mode in pvalues]
    adjusted = holm([p for _, p in main_p])
    print("\nHolm-adjusted main comparisons:")
    for (mode, p), adj in zip(main_p, adjusted):
        print(f"{mode:<38} raw_p={p:.4f} holm_p={adj:.4f}")

    for task_id, task_name in TASKS.items():
        print_cmh_table(
            f"== Task {task_name} vs pooled none baseline ==",
            filter_fn=lambda r, task_id=task_id: r["task_id"] == task_id,
            strata_fn=lambda r: r["init_id"],
        )
    for cat in ("mixed", "never", "always"):
        print_cmh_table(
            f"== Difficulty {cat} vs pooled none baseline ==",
            filter_fn=lambda r, cat=cat: category(r) == cat,
        )

    first = {(r["task_id"], r["init_id"], r["trial"]): bool(r["success"])
             for r in by_mode[("none", 1000)]}
    second = {(r["task_id"], r["init_id"], r["trial"]): bool(r["success"])
              for r in by_mode[("none", 2000)]}
    n, b, c, delta, p = mcnemar(first, second)
    placebo_group1 = by_mode[("none", 2000)]
    placebo_group0 = by_mode[("none", 1000)]
    _, expected, z, cmh_p = cmh(placebo_group1, placebo_group0, lambda r: (r["task_id"], r["init_id"]))
    print("\n== Seed placebo: none@2000 vs none@1000 ==")
    print(f"paired McNemar: n={n} b={b} c={c} delta={delta:+.1%} p={p:.4f}")
    print(f"CMH: actual={sum(r['success'] for r in placebo_group1)}/{len(placebo_group1)} "
          f"expected={expected:.2f} z={z:.3f} p={cmh_p:.4f}")

    print("\n== Color same vs different ==")
    same = [r for r in get_mode("other_init_success_same_color") if r["task_id"] in (2, 3)]
    diff = [r for r in get_mode("other_init_success_diff_color") if r["task_id"] in (2, 3)]
    if same and diff:
        obs, expected, z, p = cmh(same, diff, lambda r: (r["task_id"], r["init_id"]))
        print(f"same={int(obs)}/{len(same)} expected={expected:.2f}; "
              f"diff={sum(r['success'] for r in diff)}/{len(diff)}; z={z:.3f} p={p:.4f}")
    else:
        print("Insufficient non-empty same_color/diff_color rows.")

    color_trials = defaultdict(set)
    for row in collect:
        if row["task_id"] in (2, 3):
            color_trials[(row["task_id"], row["env_seed"])].add(row.get("target_color"))
    color_changes = [key for key, values in color_trials.items() if len(values) != 1]
    print(f"\nTarget-color reset check: {len(color_trials)} (task, env_seed) groups; "
          f"unstable={len(color_changes)}")

    print("\n== Flip rate vs none@1000 and independent-resample expectation ==")
    for mode in MODES:
        if mode == "none":
            continue
        group = {(r["task_id"], r["init_id"], r["trial"]): r for r in get_mode(mode)}
        b = c = expected_flips = 0.0
        pairs = 0
        for key, row in group.items():
            base = next((x for x in by_mode[("none", 1000)]
                         if (x["task_id"], x["init_id"], x["trial"]) == key), None)
            if base is None:
                continue
            pairs += 1
            b += bool(row["success"] and not base["success"])
            c += bool(base["success"] and not row["success"])
            p0 = sum(x["success"] for x in pooled[(key[0], key[1])]) / len(pooled[(key[0], key[1])])
            expected_flips += 2 * p0 * (1 - p0)
        print(f"{mode:<38} pairs={pairs:>4} b+c={int(b+c):>4} expected={expected_flips:.2f}")


if __name__ == "__main__":
    main()
