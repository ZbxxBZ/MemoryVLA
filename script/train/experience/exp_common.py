"""
Helpers shared by the experience-retrieval scripts in this directory (run them from the repo root with PYTHONPATH=.).

Token cache layout (cache_tokens.py; the same as the task-memory branch's cache, so an existing cache can be reused):
    <cache_dir>/index_<shard>.jsonl   one json line per episode: {"episode", "file", "T", "instruction", ...}
    <cache_dir>/<file>.pt             {"cog": [T, D_cog] fp16 raw cog tokens, "actions": [T, F+1, A] normalized chunks,
                                       "timesteps": [T], "meta": {...}}
    <cache_dir>/<file>.per.npy        [T, N, D_per] fp16 raw (compressed) perceptual tokens
For LIBERO the cache holds one frame per policy call (--frame_stride 8 --deploy_view libero).
"""

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vla.experience_retrieval import make_episode, normalize_instruction


def load_cache_index(cache_dir) -> List[dict]:
    """Episodes of a token cache, sorted by episode id; later duplicates win."""
    entries = {}
    for path in sorted(glob.glob(os.path.join(str(cache_dir), "index_*.jsonl"))):
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    entries[entry["episode"]] = entry
    return [entries[k] for k in sorted(entries)]


def task_key_of(entry: dict) -> str:
    return normalize_instruction(entry["instruction"])


def load_cache_episode(cache_dir, entry: dict, load_per: bool = True) -> dict:
    data = torch.load(Path(cache_dir) / f"{entry['file']}.pt", map_location="cpu", weights_only=False)
    out = {"cog": data["cog"], "actions": data["actions"].float(), "timesteps": data["timesteps"]}
    if load_per:
        out["per"] = torch.from_numpy(np.load(Path(cache_dir) / f"{entry['file']}.per.npy"))
    return out


def unit_keys(cog: torch.Tensor) -> torch.Tensor:
    """Retrieval keys of raw cog tokens [T, D] (as ExperienceRuntime computes them at deployment)."""
    return F.normalize(cog.float(), dim=-1).half()


def scene_signature(first_per: torch.Tensor) -> torch.Tensor:
    return F.normalize(first_per.float().flatten(), dim=0).half()


def experience_from_cache(cache_dir, entry: dict) -> dict:
    """A cached demonstration as a store episode (source `demo`, success)."""
    data = load_cache_episode(cache_dir, entry)
    return make_episode(
        task_key_of(entry), unit_keys(data["cog"]), data["actions"], scene_signature(data["per"][0]), True, "demo",
        name=f"demo/{entry['file']}", meta={"file": entry["file"], "instruction": entry["instruction"]},
    )


# === Frozen MemoryVLA parts (no VLM) ===
def read_run_config(ckpt_path) -> Tuple[dict, Path]:
    """Training config of a MemoryVLA checkpoint (`<run_dir>/checkpoints/<name>.pt`), as deploy.py reads it."""
    run_dir = Path(ckpt_path).resolve().parents[1]
    if (run_dir / "config.yaml").exists():
        import yaml

        with open(run_dir / "config.yaml", "r") as f:
            return yaml.safe_load(f) or {}, run_dir
    with open(run_dir / "config.json", "r") as f:
        return json.load(f), run_dir


def _torch_load(path, mmap: bool = False):
    """torch.load on CPU across torch versions; `mmap` avoids reading a whole 30+ GB checkpoint into RAM."""
    attempts = ([{"mmap": True, "weights_only": False}] if mmap else []) + [{"weights_only": False}, {}]
    for i, extra in enumerate(attempts):
        try:
            return torch.load(path, map_location="cpu", **extra)
        except (TypeError, RuntimeError):
            # Older torch lacks `mmap` / `weights_only`; legacy (non-zip) files cannot be memory-mapped
            if i == len(attempts) - 1:
                raise


class FrozenParts(nn.Module):
    """The parts of a MemoryVLA checkpoint that act after the VLM: episodic memory banks and the DiT action head."""

    def __init__(self, cog_mem_bank, per_mem_bank, action_model, config: dict) -> None:
        super().__init__()
        self.cog_mem_bank, self.per_mem_bank, self.action_model = cog_mem_bank, per_mem_bank, action_model
        self.config = config


def load_frozen_parts(ckpt_path, device="cuda", dtype=torch.bfloat16) -> FrozenParts:
    from action_model.action_model import ActionModel
    from vla.memory_vla import CogMemBank, PerMemBank

    cfg, _ = read_run_config(ckpt_path)
    state = _torch_load(ckpt_path, mmap=True)["model"]
    cog_state, per_state, action_state = state["cog_mem_bank"], state["per_mem_bank"], state["action_model"]
    token_size = cog_state["retrieval_blocks.0.q_proj.weight"].shape[0]
    per_token_size = per_state["retrieval_blocks.0.q_proj.weight"].shape[0]
    bank_kwargs = dict(
        dataloader_type=cfg.get("dataloader_type", "group"),
        group_size=cfg.get("group_size", 16),
        mem_length=cfg.get("mem_length", 16),
        retrieval_layers=cfg.get("retrieval_layers", 2),
        use_timestep_pe=cfg.get("use_timestep_pe", True),
        fusion_type=cfg.get("fusion_type", "gate"),
        consolidate_type=cfg.get("consolidate_type", "tome"),
        update_fused=cfg.get("update_fused", False),
    )
    cog_mem_bank = CogMemBank(token_size=token_size, **bank_kwargs)
    per_mem_bank = PerMemBank(token_size=per_token_size, **bank_kwargs)
    action_model = ActionModel(
        model_type=cfg.get("action_model_type", "DiT-L"),
        token_size=token_size,
        in_channels=cfg.get("action_dim", 7),
        future_action_window_size=cfg.get("future_action_window_size", 15),
        use_per_attn=True,
        per_token_size=per_token_size,
    )
    cog_mem_bank.load_state_dict(cog_state, strict=True)
    per_mem_bank.load_state_dict(per_state, strict=True)
    missing, unexpected = action_model.load_state_dict(action_state, strict=False)
    if missing or unexpected:
        print(f"[load_frozen_parts] action_model missing={missing} unexpected={unexpected}")
    del state

    net = action_model.net
    config = dict(
        bank_kwargs, token_size=token_size, per_token_size=per_token_size,
        action_model_type=cfg.get("action_model_type", "DiT-L"),
        future_action_window_size=cfg.get("future_action_window_size", 15), action_dim=cfg.get("action_dim", 7),
        hidden_size=net.x_embedder.linear.out_features, depth=len(net.blocks), num_heads=net.num_heads,
    )
    parts = FrozenParts(cog_mem_bank, per_mem_bank, action_model, config).to(device, dtype)
    parts.requires_grad_(False)
    return parts.eval()


@torch.no_grad()
def replay_episodic_memory(cog_mem_bank, per_mem_bank, raw_cog: torch.Tensor, raw_per: torch.Tensor):
    """
    Run the eval-mode episodic banks over the T calls of one episode in order, numbering them as the deployed model
    does (+1 per call). raw_cog [T, D_cog], raw_per [T, N, D_per] -> fused [T, D_cog], [T, N, D_per].
    """
    assert not cog_mem_bank.training and not per_mem_bank.training, "Replay needs the banks in eval mode"
    for bank in (cog_mem_bank, per_mem_bank):
        bank.reset()
    fused_cog, fused_per = [], []
    for j in range(raw_cog.shape[0]):
        fused_cog.append(cog_mem_bank.process_batch(raw_cog[j][None, None], episode_ids=[0], timesteps=[j]))
        fused_per.append(per_mem_bank.process_batch(raw_per[j][None], episode_ids=[0], timesteps=[j]))
    for bank in (cog_mem_bank, per_mem_bank):
        bank.reset()
    return torch.cat(fused_cog)[:, 0], torch.cat(fused_per)


def diffusion_loss(action_model, actions, cog, per, exp: Optional[Dict[str, torch.Tensor]] = None,
                   noise: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-sample version of `ActionModel.loss`; pass `noise` and `t` to compare conditions on identical draws."""
    if noise is None:
        noise = torch.randn_like(actions)
    if t is None:
        t = torch.randint(0, action_model.diffusion.num_timesteps, (actions.shape[0],), device=actions.device)
    x_t = action_model.diffusion.q_sample(actions, t, noise)
    pred = action_model.net(x_t, t, cog, per_token=per, exp=exp)
    return ((pred.float() - noise.float()) ** 2).mean(dim=(1, 2))
