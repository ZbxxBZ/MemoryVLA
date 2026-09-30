"""Helpers shared by the task-memory scripts in this directory (run them from the repo root with PYTHONPATH=.)."""

import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

# MIKASA-Robo task ids, as used by evaluation/mikasa/eval_mikasa.py
MIKASA_TASK_IDS = {
    "ShellGameTouch": 0,
    "InterceptMedium": 1,
    "RememberColor3": 2,
    "RememberColor5": 3,
    "RememberColor9": 4,
}
TASK_NAME_PATTERN = re.compile("|".join(sorted(MIKASA_TASK_IDS, key=len, reverse=True)))
# Prompts that identify a single task; the RememberColor prompt is shared by RC3 / RC5 / RC9
MIKASA_PROMPT_TASK_IDS = {
    "memorize the position of the cup covering the ball, then pick that cup": 0,
    "track the ball's movement, estimate its velocity, then aim the ball at the target": 1,
}


def normalize_text(text: str) -> str:
    return text.replace("’", "'").strip().lower()


def infer_task(entry: dict, task_map: Optional[Dict[str, int]] = None) -> Tuple[str, int]:
    """
    (task_key, MIKASA task id) of a cached demonstration. Checks, in order: `task_map` substrings (metadata values or
    instruction), a MIKASA environment name inside the episode metadata (e.g. a file path), then the instruction.
    An instruction shared by several tasks without identifying metadata gives task id -1.
    """
    strings = [str(v) for v in entry.get("metadata", {}).values() if v is not None]
    instruction = normalize_text(entry.get("instruction", ""))
    for pattern, task_id in (task_map or {}).items():
        if any(pattern in s for s in strings) or pattern.lower() in instruction:
            return f"task{task_id}", int(task_id)
    for s in strings:
        match = TASK_NAME_PATTERN.search(s)
        if match:
            return f"task{MIKASA_TASK_IDS[match.group(0)]}", MIKASA_TASK_IDS[match.group(0)]
    if instruction in MIKASA_PROMPT_TASK_IDS:
        task_id = MIKASA_PROMPT_TASK_IDS[instruction]
        return f"task{task_id}", task_id
    return instruction, -1


def load_task_map(path: Optional[str]) -> Optional[Dict[str, int]]:
    if not path:
        return None
    with open(path, "r") as f:
        return {k: int(v) for k, v in json.load(f).items()}


def load_cache_index(cache_dir: str) -> List[dict]:
    """Episodes of a token cache (cache_tokens.py), sorted by episode id; later duplicates win."""
    entries = {}
    for path in sorted(glob.glob(os.path.join(cache_dir, "index_*.jsonl"))):
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    entries[entry["episode"]] = entry
    return [entries[k] for k in sorted(entries)]


def flatten_metadata(tree, prefix: str = "") -> dict:
    """Per-step metadata arrays (as kept by `keep_traj_metadata`) -> {"a/b": first value}, bytes decoded."""
    if isinstance(tree, dict):
        out = {}
        for key, value in tree.items():
            out.update(flatten_metadata(value, f"{prefix}{key}/"))
        return out
    value = np.asarray(tree).reshape(-1)
    value = value[0] if value.size else None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    elif isinstance(value, np.generic):
        value = value.item()
    return {prefix.rstrip("/"): value}


def quantiles(values, qs=(0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)) -> str:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return "n=0"
    return f"n={values.size} " + " ".join(f"q{int(q * 100):02d}={np.quantile(values, q):.4f}" for q in qs)
