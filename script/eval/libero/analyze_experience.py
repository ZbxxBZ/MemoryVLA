"""
Compare LIBERO runs of the experience-retrieval branch (the *_results.jsonl files of eval_libero.py).

For every run: success rate (overall, per LIBERO-plus category when present, per trial index for online runs) and
retrieval statistics. Against --baseline, episodes are paired by (task_id, init_id, trial), which also fixes the
server seed, and compared with an exact two-sided sign test on the discordant pairs. Runs must come from the same GPU
model as the baseline: identical inputs reproduce only on the same card.

    python script/eval/libero/analyze_experience.py --baseline none \
        --runs none=logs/none_results.jsonl real=logs/real_results.jsonl unaligned=logs/unaligned_results.jsonl
"""

import argparse
import collections
import json
import math


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[(row["task_id"], row["init_id"], row["trial"])] = row  # a rerun episode replaces the earlier one
    return rows


def sign_test(a, b):
    """Two-sided exact binomial p for a vs b discordant pairs."""
    n = a + b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(a, b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def rate(rows):
    return sum(r["success"] for r in rows) / max(len(rows), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="label=path/to/*_results.jsonl")
    parser.add_argument("--baseline", default=None, help="Label of the run the others are paired against")
    args = parser.parse_args()

    runs = {}
    for item in args.runs:
        label, path = item.split("=", 1)
        runs[label] = load(path)

    for label, rows in runs.items():
        values = list(rows.values())
        print(f"\n== {label}: {len(values)} episodes, success {rate(values) * 100:.1f}%")
        by_category = collections.defaultdict(list)
        by_trial = collections.defaultdict(list)
        for row in values:
            if row.get("category") is not None:
                by_category[row["category"]].append(row)
            by_trial[row["trial"]].append(row)
        for category, group in sorted(by_category.items()):
            print(f"   {category:<28s} {rate(group) * 100:5.1f}% (n={len(group)})")
        if len(by_trial) > 1:
            print("   by trial: " + "  ".join(f"t{t}={rate(g) * 100:.1f}%" for t, g in sorted(by_trial.items())))
        infos = [row["exp_info"] for row in values if row.get("exp_info")]
        with_refs = [i for i in infos if i.get("refs")]
        if infos:
            sims = [i["mean_similarity"] for i in with_refs if "mean_similarity" in i]
            print(f"   episodes with references: {len(with_refs)}/{len(infos)}"
                  + (f", mean key similarity {sum(sims) / len(sims):.4f}" if sims else ""))
            if with_refs:
                print(f"   success with references {rate([r for r in values if (r.get('exp_info') or {}).get('refs')]) * 100:.1f}%"
                      f", without {rate([r for r in values if not (r.get('exp_info') or {}).get('refs')]) * 100:.1f}%")

    if args.baseline:
        base = runs[args.baseline]
        print(f"\n== paired against {args.baseline}")
        for label, rows in runs.items():
            if label == args.baseline:
                continue
            keys = sorted(set(rows) & set(base))
            only_run = sum(rows[k]["success"] and not base[k]["success"] for k in keys)
            only_base = sum(base[k]["success"] and not rows[k]["success"] for k in keys)
            diff = (sum(rows[k]["success"] for k in keys) - sum(base[k]["success"] for k in keys)) / max(len(keys), 1)
            print(f"   {label:<16s} pairs={len(keys):4d}  diff={diff * 100:+.1f}pt  "
                  f"discordant {only_run}/{only_base}  p={sign_test(only_run, only_base):.3f}")


if __name__ == "__main__":
    main()
