"""
Inspect an RLDS training set before caching: splits and features, episode count and lengths, instructions, and the
per-episode metadata that can tell tasks apart (the three RememberColor tasks share one instruction).

    python script/train/task_memory/inspect_dataset.py --data_root_dir /path/to/mikasa-rlds [--max_episodes 200]
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from tm_common import flatten_metadata, infer_task, load_task_map, quantiles


def to_numpy_tree(tree):
    if isinstance(tree, dict):
        return {k: to_numpy_tree(v) for k, v in tree.items()}
    return tree.numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir", required=True)
    parser.add_argument("--dataset", default="mikasa_dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_episodes", type=int, default=0, help="0 = all")
    parser.add_argument("--ckpt", default=None, help="Also print the dataset statistics saved with this checkpoint")
    parser.add_argument("--task_map", default=None, help="JSON {substring: task_id}, see tm_common.infer_task")
    args = parser.parse_args()

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds

    builder = tfds.builder(args.dataset, data_dir=args.data_root_dir)
    print(builder.info)

    if args.ckpt:
        stats_path = Path(args.ckpt).resolve().parents[1] / "dataset_statistics.json"
        with open(stats_path, "r") as f:
            stats = json.load(f)
        for name, value in stats.items():
            print(f"[checkpoint stats] {name}: num_trajectories={value.get('num_trajectories')} "
                  f"num_transitions={value.get('num_transitions')}")

    task_map = load_task_map(args.task_map)
    lengths, instructions, tasks, metadata_values = [], collections.Counter(), collections.Counter(), {}
    for i, episode in enumerate(builder.as_dataset(split=args.split, shuffle_files=False)):
        steps = list(episode["steps"])
        if i == 0:
            first = steps[0]
            print("step keys:", sorted(first.keys()))
            if "observation" in first:
                print("observation:", {k: (v.shape, v.dtype.name) for k, v in first["observation"].items()})
            print("episode keys:", sorted(k for k in episode.keys() if k != "steps"))

        instruction = steps[0]["language_instruction"].numpy().decode() if "language_instruction" in steps[0] else ""
        metadata = flatten_metadata(to_numpy_tree(episode["episode_metadata"])) if "episode_metadata" in episode else {}
        task_key, task_id = infer_task({"instruction": instruction, "metadata": metadata}, task_map)
        lengths.append(len(steps))
        instructions[instruction] += 1
        tasks[(task_key, task_id)] += 1
        for key, value in metadata.items():
            metadata_values.setdefault(key, collections.Counter())[str(value)[:120]] += 1
        if i < 3:
            print(f"episode {i}: T={len(steps)} instruction={instruction!r} metadata={metadata}")
        if args.max_episodes and i + 1 >= args.max_episodes:
            break

    print(f"\nepisodes={len(lengths)} frames={int(np.sum(lengths))} length: {quantiles(lengths)}")
    print("\ninstructions:")
    for instruction, count in instructions.most_common():
        print(f"  {count:6d}  {instruction}")
    print("\ninferred tasks (task_key, task_id; -1 = not identifiable):")
    for (task_key, task_id), count in sorted(tasks.items(), key=lambda kv: -kv[1]):
        print(f"  {count:6d}  {task_id:3d}  {task_key}")
    print("\nmetadata fields (distinct values, most common):")
    for key, counter in metadata_values.items():
        print(f"  {key}: {len(counter)} distinct; e.g. {counter.most_common(3)}")


if __name__ == "__main__":
    main()
