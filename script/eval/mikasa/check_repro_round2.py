import argparse
import glob
import json
import os


def read_rows(directory):
    rows = {}
    for path in sorted(glob.glob(os.path.join(directory, "*_results.jsonl")), key=os.path.getmtime):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["exp_mode"] == "none" and not row["record"] and row["seed_offset"] == 1000:
                    rows[(row["task_id"], row["init_id"], row["trial"])] = bool(row["success"])
    return rows


parser = argparse.ArgumentParser()
parser.add_argument("--round1", required=True)
parser.add_argument("--round2", required=True)
args = parser.parse_args()
first, second = read_rows(args.round1), read_rows(args.round2)
keys = set(first) & set(second)
if not keys:
    print("No overlapping none@1000 rows with round 1")
    raise SystemExit(0)
mismatches = sorted(key for key in keys if first[key] != second[key])
print(f"Reproducibility vs round 1: {len(keys)-len(mismatches)}/{len(keys)} identical")
if mismatches:
    print("Mismatches (task_id, init_id, trial):", mismatches)
raise SystemExit(bool(mismatches))
