"""
task_memory.py

Trained cross-episode task memory for MemoryVLA.

The working memory (the current frame's raw cog / per tokens) attends to keyframe tokens recorded in *other* episodes
of the same task, next to (not inside) MemoryVLA's episodic memory bank. The branch output is added to the episodic
fused tokens through a zero-initialized projection, so an untrained branch, or an episode without experiences, leaves
the base model unchanged.

Also holds the helpers shared by the offline scripts in `script/train/task_memory/`: loading the frozen episodic memory
and action head straight from a MemoryVLA checkpoint (without the 7B VLM), replaying the episodic memory over recorded
frames with deployment semantics, and a diffusion loss with explicit noise for paired comparisons.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """Pre-norm multi-head cross-attention followed by an FFN, both residual."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.norm_ffn = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        M, H = kv.shape[1], self.num_heads
        q = self.q_proj(self.norm_q(x)).view(B, N, H, D // H).transpose(1, 2)
        kv = self.norm_kv(kv)
        k = self.k_proj(kv).view(B, M, H, D // H).transpose(1, 2)
        v = self.v_proj(kv).view(B, M, H, D // H).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        x = x + self.o_proj(attn.transpose(1, 2).reshape(B, N, D))
        return x + self.ffn(self.norm_ffn(x))


class TaskMemoryBranch(nn.Module):
    """One token stream (cog or per): working-memory queries attend to the keyframe tokens of S experiences."""

    def __init__(
        self,
        token_dim: int,
        attn_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        max_keyframes: int = 32,
        max_slots: int = 8,
    ) -> None:
        super().__init__()
        self.q_in = nn.Linear(token_dim, attn_dim)
        self.kv_in = nn.Linear(token_dim, attn_dim)
        # Experiences are indexed by keyframe order and slot, not by absolute memory timestep: training frames carry
        # environment-step timesteps while deployment counts policy calls, so absolute times would not transfer.
        self.frame_emb = nn.Parameter(torch.randn(max_keyframes, attn_dim) * 0.02)
        self.slot_emb = nn.Parameter(torch.randn(max_slots, attn_dim) * 0.02)
        self.blocks = nn.ModuleList([CrossAttentionBlock(attn_dim, num_heads) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(attn_dim)
        self.gate = nn.Linear(2 * attn_dim, 1)
        self.out_proj = nn.Linear(attn_dim, token_dim)
        # Zero-initialized output: the branch starts as an exact no-op on top of the base model
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.collect_diag = False
        self.last_diagnostics: Dict[str, float] = {}

    def forward(self, query: torch.Tensor, exp: torch.Tensor) -> torch.Tensor:
        """
        query: [B, Nq, D] working-memory tokens; exp: [B, S, K, Nt, D] tokens of S experiences at K keyframes.
        Returns the delta [B, Nq, D] that is added to the episodic fused tokens.
        """
        B, S, K, Nt, _ = exp.shape
        kv = self.kv_in(exp) + self.frame_emb[:K][None, None, :, None, :] + self.slot_emb[:S][None, :, None, None, :]
        kv = kv.reshape(B, S * K * Nt, -1)

        q0 = self.q_in(query)
        x = q0
        for block in self.blocks:
            x = block(x, kv)
        t = self.out_norm(x)

        gate = torch.sigmoid(self.gate(torch.cat([q0, t], dim=-1)))  # [B, Nq, 1]
        delta = gate * self.out_proj(t)
        if self.collect_diag:
            self.last_diagnostics = {
                "gate": float(gate.detach().float().mean()),
                "delta_norm": float(delta.detach().float().norm(dim=-1).mean()),
            }
        return delta


class TaskMemory(nn.Module):
    """Cog and per task-memory branches; `config` is saved with the weights so the module can be rebuilt."""

    def __init__(
        self,
        cog_dim: int = 4096,
        per_dim: int = 256,
        cog_attn_dim: int = 1024,
        per_attn_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        max_keyframes: int = 32,
        max_slots: int = 8,
    ) -> None:
        super().__init__()
        self.config = dict(
            cog_dim=cog_dim, per_dim=per_dim, cog_attn_dim=cog_attn_dim, per_attn_dim=per_attn_dim,
            num_layers=num_layers, num_heads=num_heads, max_keyframes=max_keyframes, max_slots=max_slots,
        )
        self.cog = TaskMemoryBranch(cog_dim, cog_attn_dim, num_layers, num_heads, max_keyframes, max_slots)
        self.per = TaskMemoryBranch(per_dim, per_attn_dim, num_layers, num_heads, max_keyframes, max_slots)

    def forward(
        self,
        cog_query: torch.Tensor,  # [B, 1, D_cog]
        per_query: torch.Tensor,  # [B, N, D_per]
        cog_exp: torch.Tensor,  # [B, S, K, D_cog]
        per_exp: torch.Tensor,  # [B, S, K, N, D_per]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cog(cog_query, cog_exp.unsqueeze(3)), self.per(per_query, per_exp)

    def set_collect_diag(self, enabled: bool) -> None:
        self.cog.collect_diag = self.per.collect_diag = enabled

    @property
    def last_diagnostics(self) -> Dict[str, float]:
        return {
            **{f"task_cog_{k}": v for k, v in self.cog.last_diagnostics.items()},
            **{f"task_per_{k}": v for k, v in self.per.last_diagnostics.items()},
        }


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


def save_task_memory(module: TaskMemory, path, **extra) -> None:
    torch.save({"config": module.config, "state_dict": module.state_dict(), **extra}, path)


def load_task_memory(path) -> Tuple[TaskMemory, dict]:
    ckpt = _torch_load(path)
    module = TaskMemory(**ckpt["config"])
    module.load_state_dict(ckpt["state_dict"])
    return module, ckpt


def read_run_config(ckpt_path) -> Tuple[dict, Path]:
    """Training config of a MemoryVLA checkpoint (`<run_dir>/checkpoints/<name>.pt`), as used by deploy.py."""
    run_dir = Path(ckpt_path).resolve().parents[1]
    if (run_dir / "config.yaml").exists():
        import yaml

        with open(run_dir / "config.yaml", "r") as f:
            return yaml.safe_load(f) or {}, run_dir
    with open(run_dir / "config.json", "r") as f:
        return json.load(f), run_dir


class FrozenMemoryVLAParts(nn.Module):
    """The parts of a MemoryVLA checkpoint that act after the VLM: episodic memory banks and the DiT action head."""

    def __init__(self, cog_mem_bank, per_mem_bank, action_model, config: dict) -> None:
        super().__init__()
        self.cog_mem_bank, self.per_mem_bank, self.action_model = cog_mem_bank, per_mem_bank, action_model
        self.config = config


def load_frozen_parts(ckpt_path, device="cuda", dtype=torch.bfloat16) -> FrozenMemoryVLAParts:
    """Build CogMemBank / PerMemBank / ActionModel with the checkpoint's config and load their weights, skipping the VLM."""
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

    config = dict(
        bank_kwargs, token_size=token_size, per_token_size=per_token_size,
        action_model_type=cfg.get("action_model_type", "DiT-L"),
        future_action_window_size=cfg.get("future_action_window_size", 15), action_dim=cfg.get("action_dim", 7),
    )
    parts = FrozenMemoryVLAParts(cog_mem_bank, per_mem_bank, action_model, config).to(device, dtype)
    parts.requires_grad_(False)
    return parts.eval()


@torch.no_grad()
def replay_episodic_memory(
    cog_mem_bank,
    per_mem_bank,
    raw_cog: torch.Tensor,  # [B, T, D_cog]
    raw_per: torch.Tensor,  # [B, T, N, D_per]
    timesteps: torch.Tensor,  # [B, T] memory timesteps
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the (eval-mode) episodic banks over T frames of B independent episodes in order, exactly as consecutive
    `predict_action` calls do. Returns the fused tokens: [B, T, D_cog] and [B, T, N, D_per].
    """
    assert not cog_mem_bank.training and not per_mem_bank.training, "Replay needs the banks in eval mode"
    B, T = raw_cog.shape[:2]
    episode_ids = list(range(B))
    for bank in (cog_mem_bank, per_mem_bank):
        bank.reset()
        bank.clear_pinned()
    fused_cog, fused_per = [], []
    for j in range(T):
        steps = [timesteps[b, j] for b in range(B)]
        fused_cog.append(cog_mem_bank.process_batch(raw_cog[:, j : j + 1], episode_ids=episode_ids, timesteps=steps))
        fused_per.append(per_mem_bank.process_batch(raw_per[:, j], episode_ids=episode_ids, timesteps=steps))
    for bank in (cog_mem_bank, per_mem_bank):
        bank.reset()
    return torch.stack(fused_cog, dim=1)[:, :, 0], torch.stack(fused_per, dim=1)


def diffusion_loss(
    action_model,
    actions: torch.Tensor,  # [B, T, A] normalized action chunk
    cog: torch.Tensor,  # [B, 1, D_cog]
    per: torch.Tensor,  # [B, N, D_per]
    noise: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-sample version of `ActionModel.loss`; pass `noise` and `t` to compare conditions on identical draws."""
    if noise is None:
        noise = torch.randn_like(actions)
    if t is None:
        t = torch.randint(0, action_model.diffusion.num_timesteps, (actions.shape[0],), device=actions.device)
    x_t = action_model.diffusion.q_sample(actions, t, noise)
    pred = action_model.net(x_t, t, cog, per_token=per)
    return ((pred.float() - noise.float()) ** 2).mean(dim=(1, 2))


def scene_signature(first_per: torch.Tensor) -> torch.Tensor:
    """Unit vector of a frame's per tokens; same-scene recordings have a first-frame cosine similarity near 1."""
    return F.normalize(first_per.reshape(-1).float(), dim=0)


def uniform_keyframes(num_steps: int, num_keyframes: int) -> List[int]:
    from vla.experience import select_keyframes

    return select_keyframes(num_steps, num_keyframes, "uniform")


def stack_padded(indices: Sequence[int], num: int) -> List[int]:
    """Pad a keyframe index list to `num` entries by repeating the last one (episodes shorter than K calls)."""
    indices = list(indices)
    return indices + [indices[-1]] * (num - len(indices))
