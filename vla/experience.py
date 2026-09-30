"""
experience.py

Cross-episode experience store used to prefill (pin) MemoryVLA's memory banks at the start of an episode.

Each experience is one recorded episode (one entry per `predict_action` call):
    <name>.pt      -> {"meta", "cog" [T, D_cog] fp16, "actions" [T, H, A], "timesteps" [T]}
    <name>.per.pt  -> {"per" [T, N, D_per] fp16}  (kept separate so retrieval only loads the light file)
and one line in `index.jsonl` holding its metadata.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

EXP_MODES = (
    "none",
    "other_init_success",
    "same_init_success",
    "same_init_failure",
    "other_task_success",
    "noise",
    "other_init_success_same_color",
    "other_init_success_diff_color",
)
KEYFRAME_STRATEGIES = ("uniform", "event")
PIN_TIMESTEP_MODES = ("orig", "neg", "zero")
PIN_TARGETS = ("both", "cog", "per")


def _debounce(binary: np.ndarray, min_run: int) -> np.ndarray:
    """Absorb runs shorter than `min_run` into the preceding run, so gripper chatter is not counted as a toggle."""
    out = binary.copy()
    start = 0
    for i in range(1, len(binary) + 1):
        if i == len(binary) or binary[i] != binary[start]:
            if start > 0 and i - start < min_run:
                out[start:i] = out[start - 1]
            start = i
    return out


def select_keyframes(
    num_steps: int,
    num_keyframes: int,
    strategy: str = "uniform",
    actions: Optional[np.ndarray] = None,
    min_run: int = 2,
) -> List[int]:
    """Pick keyframe indices of an episode; `actions` is [T, H, A] (normalized chunks, gripper in the last dim)."""
    assert strategy in KEYFRAME_STRATEGIES, f"Unknown keyframe strategy `{strategy}`"
    assert num_keyframes >= 2, "Need at least the first and last frame"
    if num_steps <= num_keyframes:
        return list(range(num_steps))

    if strategy == "uniform":
        return np.linspace(0, num_steps - 1, num_keyframes).round().astype(int).tolist()

    # event: first + last frame, the step right before each (debounced) gripper toggle, then farthest-point fill
    picks = {0, num_steps - 1}
    gripper = _debounce(actions[:, 0, -1] > 0.5, min_run)
    for t in np.nonzero(gripper[1:] != gripper[:-1])[0] + 1:
        picks.add(max(int(t) - 1, 0))

    picks = sorted(picks)
    if len(picks) > num_keyframes:
        middle = picks[1:-1]
        keep = np.linspace(0, len(middle) - 1, num_keyframes - 2).round().astype(int)
        picks = [picks[0]] + [middle[i] for i in keep] + [picks[-1]]

    while len(picks) < num_keyframes:
        candidates = [t for t in range(num_steps) if t not in picks]
        gaps = [min(abs(t - p) for p in picks) for t in candidates]
        picks.append(candidates[int(np.argmax(gaps))])

    return sorted(picks)


def pin_timesteps(timesteps: np.ndarray, mode: str) -> np.ndarray:
    """Map an experience's recorded memory timesteps to the ones used when pinning it into a new episode."""
    assert mode in PIN_TIMESTEP_MODES, f"Unknown pin timestep mode `{mode}`"
    timesteps = np.asarray(timesteps)
    if mode == "orig":
        return timesteps
    if mode == "neg":
        # shift the whole experience before the current episode's step 0, keeping relative spacing
        return timesteps - timesteps.max() - 1
    return np.zeros_like(timesteps)


class ExperienceStore:
    def __init__(self, root: str) -> None:
        self.root = root
        self.index_path = os.path.join(root, "index.jsonl")
        os.makedirs(root, exist_ok=True)

        # name -> meta (later lines win, so re-recorded episodes replace older ones)
        self.index: Dict[str, dict] = {}
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                for line in f:
                    if line.strip():
                        meta = json.loads(line)
                        self.index[meta["name"]] = meta
        self._cache: Dict[str, dict] = {}
        self._noise_stats: Dict[tuple, tuple] = {}
        print(f"*** ExperienceStore at `{root}` with {len(self.index)} experiences ***")

    def save(self, meta: dict, cog: torch.Tensor, per: torch.Tensor, actions: np.ndarray, timesteps: np.ndarray) -> str:
        outcome = "succ" if meta["success"] else "fail"
        name = (
            f"{meta['suite']}_task{meta['task_id']}_init{meta['init_id']}"
            f"_trial{meta['trial']}_seed{meta['seed']}_{outcome}"
        )
        rel_dir = os.path.join(str(meta["suite"]), f"task{meta['task_id']}")
        os.makedirs(os.path.join(self.root, rel_dir), exist_ok=True)
        meta = {**meta, "name": name, "path": os.path.join(rel_dir, name)}

        base = os.path.join(self.root, meta["path"])
        torch.save(
            {"meta": meta, "cog": cog.half().cpu(), "actions": np.asarray(actions), "timesteps": np.asarray(timesteps)},
            base + ".pt",
        )
        torch.save({"per": per.half().cpu()}, base + ".per.pt")

        with open(self.index_path, "a") as f:
            f.write(json.dumps(meta) + "\n")
        self.index[name] = meta
        self._cache.pop(name, None)
        return base + ".pt"

    def load(self, meta: dict) -> dict:
        if meta["name"] not in self._cache:
            self._cache[meta["name"]] = torch.load(os.path.join(self.root, meta["path"] + ".pt"), map_location="cpu")
        return self._cache[meta["name"]]

    def load_per(self, meta: dict) -> torch.Tensor:
        return torch.load(os.path.join(self.root, meta["path"] + ".per.pt"), map_location="cpu")["per"]

    def query(
        self,
        mode: str,
        suite: str,
        task_id: int,
        init_id: int,
        k: int,
        seed: int = 0,
        cur_meta: Optional[dict] = None,
    ) -> List[dict]:
        """Return up to `k` experience records for the current episode according to `mode`."""
        assert mode in EXP_MODES, f"Unknown experience mode `{mode}`"
        if mode == "none" or k <= 0:
            return []

        want_success = mode != "same_init_failure"
        same_task = [
            m for m in self.index.values()
            if m["suite"] == suite and m["task_id"] == task_id and m.get("exp_mode", "none") == "none"
        ]
        all_suite = [m for m in self.index.values() if m["suite"] == suite and m.get("exp_mode", "none") == "none"]
        if mode == "other_task_success":
            candidates = [m for m in all_suite if m["task_id"] != task_id and m["success"]]
        elif mode == "noise":
            candidates = same_task
        elif mode in ("other_init_success_same_color", "other_init_success_diff_color"):
            color = (cur_meta or {}).get("target_color")
            candidates = [
                m for m in same_task
                if m["success"] and m["init_id"] != init_id and m.get("target_color") is not None
                and color is not None
                and ((m["target_color"] == color) == (mode == "other_init_success_same_color"))
            ]
        else:
            want_same_init = mode != "other_init_success"
            candidates = [
                m for m in same_task
                if m["success"] == want_success and (m["init_id"] == init_id) == want_same_init
            ]
        if len(candidates) == 0:
            print(f"[ExperienceStore] no `{mode}` experience for suite={suite} task={task_id} init={init_id}")
            return []
        candidates = sorted(candidates, key=lambda m: m["name"])

        # The original B mode retains first-frame similarity ranking. New placebo and color
        # conditions use seeded random selection to avoid adding a similarity variable.
        key_meta = next((m for m in same_task if m["init_id"] == init_id), None)
        if mode == "other_init_success" and key_meta is not None and len(candidates) > k:
            key = self.load(key_meta)["cog"][0].float()
            keys = torch.stack([self.load(m)["cog"][0].float() for m in candidates])
            order = F.cosine_similarity(keys, key[None], dim=-1).argsort(descending=True)[:k].tolist()
        else:
            rng = np.random.RandomState(seed)
            order = rng.permutation(len(candidates))[:k].tolist()

        selected = [candidates[i] for i in order]
        if mode == "noise":
            return [{"name": f"noise_from_{selected[0]['name']}", "noise": True, "source": selected[0]}]
        return selected

    def _get_noise_stats(self, suite: str, task_id: int):
        key = (suite, task_id)
        if key not in self._noise_stats:
            candidates = [m for m in self.index.values()
                          if m["suite"] == suite and m["task_id"] == task_id
                          and m.get("exp_mode", "none") == "none"]
            cog_count = per_count = 0
            cog_sum = cog_sq = per_sum = per_sq = None
            for meta in candidates:
                record = self.load(meta)
                cog = record["cog"].float().reshape(-1, record["cog"].shape[-1])
                per_tokens = self.load_per(meta).float()
                per = per_tokens.reshape(-1, per_tokens.shape[-1])
                if cog_sum is None:
                    cog_sum, cog_sq = cog.sum(0), cog.square().sum(0)
                    per_sum, per_sq = per.sum(0), per.square().sum(0)
                else:
                    cog_sum += cog.sum(0)
                    cog_sq += cog.square().sum(0)
                    per_sum += per.sum(0)
                    per_sq += per.square().sum(0)
                cog_count += cog.shape[0]
                per_count += per.shape[0]
            if not candidates or not cog_count or not per_count:
                raise ValueError(f"Cannot synthesize noise without stored tokens for {suite} task {task_id}")
            cog_mean, cog_var = cog_sum / cog_count, (cog_sq / cog_count - (cog_sum / cog_count).square()).clamp_min(0)
            per_mean, per_var = per_sum / per_count, (per_sq / per_count - (per_sum / per_count).square()).clamp_min(0)
            self._noise_stats[key] = (cog_mean, cog_var.sqrt(), per_mean, per_var.sqrt())
        return self._noise_stats[key]

    def build_pinned_entries(
        self,
        metas: List[dict],
        num_keyframes: int,
        strategy: str,
        timestep_mode: str,
        target: str,
        device: torch.device,
        cog_dtype: torch.dtype,
        per_dtype: torch.dtype,
        seed: int = 0,
    ):
        """Turn experiences into `(timestep, feat[N, D])` entries for `CogMemBank.set_pinned` / `PerMemBank.set_pinned`."""
        assert target in PIN_TARGETS, f"Unknown pin target `{target}`"
        cog_entries, per_entries = [], []
        noise_generator = torch.Generator(device="cpu").manual_seed(int(seed))
        for meta in metas:
            is_noise = bool(meta.get("noise"))
            source = meta["source"] if is_noise else meta
            record = self.load(source)
            idx = select_keyframes(record["cog"].shape[0], num_keyframes, strategy, record["actions"])
            timesteps = pin_timesteps(record["timesteps"], timestep_mode)
            per = self.load_per(source) if target in ("both", "per") else None
            if is_noise:
                cog_mean, cog_std, per_mean, per_std = self._get_noise_stats(source["suite"], source["task_id"])
            for i in idx:
                t = torch.tensor(int(timesteps[i]), device=device)
                if target in ("both", "cog"):
                    if is_noise:
                        shape = record["cog"][i].shape
                        feat = cog_mean + torch.randn(shape, generator=noise_generator) * cog_std
                    else:
                        feat = record["cog"][i]
                    cog_entries.append((t, feat.to(device, cog_dtype)[None]))
                if per is not None:
                    if is_noise:
                        shape = per[i].shape
                        feat = per_mean + torch.randn(shape, generator=noise_generator) * per_std
                    else:
                        feat = per[i]
                    per_entries.append((t, feat.to(device, per_dtype)))
        return cog_entries, per_entries
