"""
Build a demo experience library (an ExperienceStore file for deploy.py --exp_library) from a token cache.

Each cached demonstration becomes one successful `demo` episode keyed by its normalized instruction, with per-call
retrieval keys (unit raw cog tokens), action chunks and a first-frame signature. No model is needed.

    PYTHONPATH=. python script/train/experience/build_library.py --cache_dir ./cache/experience/goal_tokens \
        --out ./cache/experience/goal_library.pt
"""

import argparse
import collections

from exp_common import experience_from_cache, load_cache_index
from vla.experience_retrieval import ExperienceStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    entries = load_cache_index(args.cache_dir)
    store = ExperienceStore([experience_from_cache(args.cache_dir, entry) for entry in entries])
    lengths = collections.defaultdict(list)
    for episode in store.episodes:
        lengths[episode["task_key"]].append(episode["keys"].shape[0])
    for task, values in sorted(lengths.items()):
        print(f"{len(values):4d} demos, {min(values)}-{max(values)} calls: {task}")
    store.save(args.out)
    print(f"saved {len(store)} episodes of {len(lengths)} tasks to {args.out}")


if __name__ == "__main__":
    main()
