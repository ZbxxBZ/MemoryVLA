"""
experience_retrieval.py

Cross-episode experience retrieval for MemoryVLA: an action-centric episodic memory read by the DiT action head.

Every stored episode keeps, per policy call, a retrieval key (the call's raw cognitive token, unit-normalized) and the
normalized action chunk executed from that call on, plus its outcome and a first-frame scene signature. For the current
episode a few stored episodes of the same task are picked as references. Each reference carries a pointer that only
moves forward: at every call it advances to the most similar key among the next `window` calls, so it stays at the
matching phase of the task even where frames of different phases look alike. What the reference did from there (its
action chunk), where it is (its progress, the similarity, call indices) and how it ended (success) become one token per
reference. The action head reads these tokens through tanh-gated cross-attention layers inserted after its blocks; the
gates start at zero and an episode without references skips the layers, so the base policy is unchanged until the
adapter is trained and given references.

MemoryVLA's episodic memory banks are left untouched: other episodes never enter them.
"""

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Call indices are divided by this before entering the adapter (LIBERO episodes are at most ~70 policy calls)
CALL_SCALE = 64.0
SCALAR_NAMES = ("ref_progress", "similarity", "call", "ref_call", "ref_length")
CONDITIONS = ("none", "real", "unaligned")


def normalize_instruction(text: str) -> str:
    """Task key of an instruction: lowercase, typographic apostrophes folded, whitespace collapsed."""
    return " ".join(text.replace("’", "'").strip().lower().split())


def reference_scalars(j: int, similarity: float, call: int, ref_length: int) -> List[float]:
    """Scalar features of a reference at its step `j` while the current episode is at call `call` (SCALAR_NAMES)."""
    return [j / max(ref_length - 1, 1), similarity, call / CALL_SCALE, j / CALL_SCALE, ref_length / CALL_SCALE]


# === Alignment ===
def pointer_align(sims: np.ndarray, window: int) -> np.ndarray:
    """
    Offline version of `PointerTracker`: sims [T_query, T_ref] (query calls x reference calls) -> the reference step
    [T_query] matched at every query call. The pointer starts at 0 and, at each call, moves to the most similar step
    among itself and the next `window` steps.
    """
    T_q, T_r = sims.shape
    js = np.empty(T_q, dtype=np.int64)
    ptr = 0
    for t in range(T_q):
        hi = min(ptr + window, T_r - 1)
        ptr = ptr + int(np.argmax(sims[t, ptr : hi + 1]))
        js[t] = ptr
    return js


class PointerTracker:
    """Online forward-only alignment of one reference to the running episode (same rule as `pointer_align`)."""

    def __init__(self, keys: torch.Tensor, window: int) -> None:
        self.keys, self.window, self.ptr = keys, window, 0  # keys: [T_ref, D] unit vectors

    def step(self, query: torch.Tensor) -> Tuple[int, float]:
        hi = min(self.ptr + self.window, self.keys.shape[0] - 1)
        sims = self.keys[self.ptr : hi + 1].float() @ query.float()
        k = int(torch.argmax(sims))
        self.ptr += k
        return self.ptr, float(sims[k])


# === Adapter ===
class GatedCrossAttention(nn.Module):
    """Action tokens read the reference tokens; the residual is scaled by tanh(alpha), alpha initialized to 0."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.o_proj = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, tokens, attend, active):
        """x [B, N, D]; tokens [B, M, D]; attend [B, M] bool (True = may be attended); active [B]."""
        B, N, D = x.shape
        M, H = tokens.shape[1], self.num_heads
        q = self.q_proj(self.norm(x)).view(B, N, H, D // H).transpose(1, 2)
        k, v = self.kv_proj(tokens).view(B, M, 2, H, D // H).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=attend[:, None, None, :])
        out = self.o_proj(out.transpose(1, 2).reshape(B, N, D))
        scale = torch.tanh(self.alpha).to(out.dtype) * active.to(out.dtype)  # episodes without references: exactly 0
        return x + (scale[:, None, None] * out).to(x.dtype)


class ExperienceAdapter(nn.Module):
    """
    Encodes M references into tokens and injects them into the DiT through gated cross-attention after the blocks
    listed in `inject_layers`. The input dict (built by `ReferenceBatch` / `ExperienceRuntime`) holds, for B samples:
        actions [B, M, F, A]  reference action chunk at the matched step (normalized as in training)
        scalars [B, M, S]     `reference_scalars`
        success [B, M]        outcome of the reference episode (1 = success)
        valid   [B, M]        which slots hold a reference
    A learned null token is always attendable, so a sample with few references never attends to an empty set.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        depth: int = 24,
        inject_every: int = 2,
        action_dim: int = 7,
        chunk_len: int = 16,
        max_refs: int = 8,
    ) -> None:
        super().__init__()
        self.config = dict(
            hidden_size=hidden_size, num_heads=num_heads, depth=depth, inject_every=inject_every,
            action_dim=action_dim, chunk_len=chunk_len, max_refs=max_refs,
        )
        self.inject_layers = [i for i in range(depth) if (i + 1) % inject_every == 0]
        self.action_mlp = nn.Sequential(
            nn.Linear(chunk_len * action_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.scalar_mlp = nn.Sequential(
            nn.Linear(len(SCALAR_NAMES), hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.success_emb = nn.Embedding(2, hidden_size)
        self.slot_emb = nn.Parameter(torch.randn(max_refs, hidden_size) * 0.02)
        self.null_token = nn.Parameter(torch.randn(1, hidden_size) * 0.02)
        self.token_norm = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleDict({str(i): GatedCrossAttention(hidden_size, num_heads) for i in self.inject_layers})

    def encode(self, exp: Dict[str, torch.Tensor]):
        actions = exp["actions"]
        B, M = actions.shape[:2]
        assert M <= self.slot_emb.shape[0], f"{M} references > max_refs {self.slot_emb.shape[0]}"
        dtype = self.null_token.dtype
        h = (
            self.action_mlp(actions.flatten(2).to(dtype))
            + self.scalar_mlp(exp["scalars"].to(dtype))
            + self.success_emb(exp["success"].long())
        )
        h = self.token_norm(h) + self.slot_emb[:M]
        tokens = torch.cat([self.null_token.expand(B, 1, -1).to(h.dtype), h], dim=1)  # [B, 1 + M, H]
        valid = exp["valid"].bool()
        attend = torch.cat([torch.ones_like(valid[:, :1]), valid], dim=1)  # the null token is always attendable
        active = valid.any(dim=1)
        return tokens, attend, active

    def inject(self, block_index: int, x: torch.Tensor, encoded) -> torch.Tensor:
        key = str(block_index)
        if key not in self.layers:
            return x
        tokens, attend, active = encoded
        return self.layers[key](x, tokens, attend, active)

    def gates(self) -> Dict[str, float]:
        return {f"gate_{k}": float(torch.tanh(layer.alpha.detach().float())) for k, layer in self.layers.items()}


def save_adapter(adapter: ExperienceAdapter, path, **extra) -> None:
    torch.save({"config": adapter.config, "state_dict": adapter.state_dict(), **extra}, path)


def load_adapter(path) -> Tuple[ExperienceAdapter, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    adapter = ExperienceAdapter(**ckpt["config"])
    adapter.load_state_dict(ckpt["state_dict"])
    return adapter, ckpt


# === Store ===
def make_episode(
    task_key: str,
    keys: torch.Tensor,  # [T, D] unit vectors
    actions: torch.Tensor,  # [T, F, A] normalized action chunks
    sig: Optional[torch.Tensor],  # [Ds] unit first-frame signature
    success: bool,
    source: str,
    name: str,
    meta: Optional[dict] = None,
) -> dict:
    assert keys.shape[0] == actions.shape[0], f"{keys.shape[0]} keys vs {actions.shape[0]} action chunks"
    return {
        "name": name, "task_key": task_key, "keys": keys.half().cpu(), "actions": actions.float().cpu(),
        "sig": None if sig is None else sig.half().cpu(), "success": bool(success), "source": source,
        "meta": dict(meta or {}),
    }


class ExperienceStore:
    """Episodes grouped by task key. Episodes are never modified once added, so stores can share them."""

    def __init__(self, episodes: Optional[Sequence[dict]] = None) -> None:
        self.episodes: List[dict] = []
        self.by_task: Dict[str, List[int]] = {}
        for episode in episodes or []:
            self.add(episode)

    def add(self, episode: dict) -> int:
        index = len(self.episodes)
        self.episodes.append(episode)
        self.by_task.setdefault(episode["task_key"], []).append(index)
        return index

    def copy(self) -> "ExperienceStore":
        return ExperienceStore(self.episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def candidates(self, task_key: str, sources=("demo", "rollout"), success_only: bool = True,
                   exclude: Sequence[str] = ()) -> List[int]:
        exclude = set(exclude)
        return [
            i for i in self.by_task.get(task_key, [])
            if self.episodes[i]["source"] in sources and (self.episodes[i]["success"] or not success_only)
            and self.episodes[i]["name"] not in exclude
        ]

    def select(self, task_key: str, k: int, strategy: str = "similar", sig: Optional[torch.Tensor] = None,
               rng: Optional[np.random.Generator] = None, **filters) -> List[int]:
        """
        Up to k episodes of the task. `similar`: highest first-frame signature similarity (falls back to `recent`
        without signatures); `recent`: the last added; `random`: uniform without replacement.
        """
        pool = self.candidates(task_key, **filters)
        if not pool or k <= 0:
            return []
        if strategy == "similar" and sig is not None and all(self.episodes[i]["sig"] is not None for i in pool):
            sigs = torch.stack([self.episodes[i]["sig"] for i in pool]).float()
            order = torch.argsort(sigs @ sig.float().cpu(), descending=True).tolist()
            return [pool[o] for o in order[:k]]
        if strategy == "random":
            rng = rng or np.random.default_rng()
            return [pool[i] for i in rng.permutation(len(pool))[:k]]
        return pool[-k:][::-1]

    def save(self, path) -> None:
        torch.save({"version": 1, "episodes": self.episodes}, path)

    @classmethod
    def load(cls, paths: Sequence) -> "ExperienceStore":
        store = cls()
        for path in paths:
            for episode in torch.load(path, map_location="cpu", weights_only=False)["episodes"]:
                store.add(episode)
        return store

    def summary(self) -> Dict[str, Dict[str, int]]:
        out = {}
        for task, indices in self.by_task.items():
            counts = {}
            for i in indices:
                ep = self.episodes[i]
                tag = f"{ep['source']}_{'success' if ep['success'] else 'failure'}"
                counts[tag] = counts.get(tag, 0) + 1
            out[task] = counts
        return out


# === Reference features ===
def reference_slot(ref: dict, j: int, similarity: float, call: int) -> Tuple[torch.Tensor, List[float], int]:
    T = ref["actions"].shape[0]
    return ref["actions"][j], reference_scalars(j, similarity, call, T), int(ref["success"])


def stack_slots(slots: List[Tuple[torch.Tensor, List[float], int]], max_refs: int, chunk_shape) -> Dict[str, torch.Tensor]:
    """One sample's references -> padded [1, max_refs, ...] tensors (CPU)."""
    actions = torch.zeros(1, max_refs, *chunk_shape)
    scalars = torch.zeros(1, max_refs, len(SCALAR_NAMES))
    success = torch.zeros(1, max_refs, dtype=torch.long)
    valid = torch.zeros(1, max_refs, dtype=torch.bool)
    for m, (chunk, scal, succ) in enumerate(slots[:max_refs]):
        actions[0, m], scalars[0, m], success[0, m], valid[0, m] = chunk, torch.tensor(scal), succ, True
    return {"actions": actions, "scalars": scalars, "success": success, "valid": valid}


def executed_chunks(predicted: List[np.ndarray], executed_per_call: int, chunk_len: int) -> np.ndarray:
    """
    Per-call chunks of what a rollout executed: call k's chunk is the first `executed_per_call` predicted actions of
    calls k, k+1, ... concatenated; past the last executed action it continues with the last call's unexecuted
    prediction, then repeats the final row. predicted: list of [F, A] normalized chunks -> [T, chunk_len, A].
    """
    executed = np.concatenate([p[:executed_per_call] for p in predicted], axis=0)
    tail = np.concatenate([executed, predicted[-1][executed_per_call:]], axis=0)
    out = []
    for k in range(len(predicted)):
        chunk = tail[k * executed_per_call : k * executed_per_call + chunk_len]
        if len(chunk) < chunk_len:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], chunk_len - len(chunk), axis=0)], axis=0)
        out.append(chunk)
    return np.stack(out).astype(np.float32)


class ExperienceRuntime:
    """
    Server-side state of one episode: picks the references on the first call (when the first-frame signature is
    known), steps their pointers at every call, hands the action head their features, and records the episode
    (keys, executed chunks, signature) so it can be added to a store at the end.
    """

    def __init__(
        self,
        store: ExperienceStore,
        task_key: str,
        condition: str = "real",
        num_refs: int = 4,
        window: int = 3,
        select: str = "similar",
        sources=("demo", "rollout"),
        max_refs: int = 8,
        chunk_shape=(16, 7),
        seed: int = 0,
        device="cuda",
    ) -> None:
        assert condition in CONDITIONS, f"Unknown condition {condition}"
        self.store, self.task_key, self.condition = store, task_key, condition
        self.num_refs, self.window, self.select, self.sources = min(num_refs, max_refs), window, select, tuple(sources)
        self.max_refs, self.chunk_shape, self.device = max_refs, tuple(chunk_shape), device
        self.rng = np.random.default_rng(seed)
        self.refs: List[dict] = []
        self.trackers: List[PointerTracker] = []
        self.call = 0
        self.keys: List[torch.Tensor] = []
        self.predicted: List[np.ndarray] = []
        self.sig: Optional[torch.Tensor] = None
        self.trace: List[List[Tuple[int, float]]] = []  # per call: (j, similarity) per reference

    def __call__(self, raw_cog: torch.Tensor, raw_per: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        key = F.normalize(raw_cog.float().flatten(), dim=0)
        if self.call == 0:
            self.sig = F.normalize(raw_per.float().flatten(), dim=0).half().cpu()
            if self.condition != "none":
                picked = self.store.select(self.task_key, self.num_refs, self.select, sig=self.sig, rng=self.rng,
                                           sources=self.sources)
                self.refs = [self.store.episodes[i] for i in picked]
                self.trackers = [PointerTracker(ref["keys"].to(self.device), self.window) for ref in self.refs]
        self.keys.append(key.half().cpu())
        exp = None
        if self.refs:
            slots, step_trace = [], []
            for ref, tracker in zip(self.refs, self.trackers):
                j, sim = tracker.step(key)
                if self.condition == "unaligned":  # control: a random step of the same reference
                    j = int(self.rng.integers(ref["keys"].shape[0]))
                    sim = float(ref["keys"][j].float() @ key.cpu())
                slots.append(reference_slot(ref, j, sim, self.call))
                step_trace.append((j, sim))
            self.trace.append(step_trace)
            exp = {k: v.to(self.device) for k, v in stack_slots(slots, self.max_refs, self.chunk_shape).items()}
        self.call += 1
        return exp

    def record_prediction(self, normalized_chunk: np.ndarray) -> None:
        self.predicted.append(np.asarray(normalized_chunk, dtype=np.float32).copy())

    def info(self) -> dict:
        out = {"refs": [ref["name"] for ref in self.refs], "num_calls": self.call, "condition": self.condition}
        if self.trace:
            sims = np.array([[s for _, s in step] for step in self.trace])
            out["mean_similarity"] = float(sims.mean())
            out["final_ref_progress"] = [
                float(self.trace[-1][m][0] / max(ref["keys"].shape[0] - 1, 1)) for m, ref in enumerate(self.refs)
            ]
        return out

    def to_episode(self, success: bool, executed_per_call: int, name: Optional[str] = None,
                   meta: Optional[dict] = None) -> Optional[dict]:
        if not self.predicted or len(self.predicted) != len(self.keys):
            return None
        chunks = executed_chunks(self.predicted, executed_per_call, self.chunk_shape[0])
        return make_episode(
            self.task_key, torch.stack(self.keys), torch.from_numpy(chunks), self.sig, success, "rollout",
            name or f"rollout_{time.time_ns()}", meta,
        )
