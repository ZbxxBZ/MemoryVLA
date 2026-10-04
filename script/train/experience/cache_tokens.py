"""
Cache MemoryVLA's frozen raw tokens for every policy call of an RLDS training set (default: LIBERO-Goal).

Per episode this writes, under --out_dir (layout in exp_common.py, shared with the task-memory branch's cache):
    <file>.pt        {"cog": [T, D_cog] fp16, "actions": [T, F+1, A] fp32 (normalized as in training),
                      "timesteps": [T] int64 (call index), "env_steps": [T] int64, "meta": {...}}
    <file>.per.npy   [T, N, D_per] fp16
plus one json line per episode in index_<shard>.jsonl. Frames go through the training RLDS pipeline without
augmentation; with the defaults one frame is kept per policy call (8 env steps, as the LIBERO evaluation executes
8 actions per call) and preprocessed as the evaluation client and server do (libero_deploy_view). Tokens are extracted
as in `MemoryVLA.predict_action`. Finished episodes are skipped, so an interrupted shard can be restarted.

    PYTHONPATH=. python script/train/experience/cache_tokens.py --ckpt /path/to/run/checkpoints/x.pt \
        --data_root_dir /path/to/libero-rlds --dataset libero_goal_no_noops --out_dir ./cache/experience/goal_tokens
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from exp_common import read_run_config

LOAD_KEYS_SKIPPED = ("model_id_or_path", "saved_model_path", "pretrained_checkpoint")


def libero_deploy_view(image: np.ndarray) -> Image.Image:
    """
    The image a LIBERO evaluation feeds the policy, from a dataset frame (already in the training orientation):
    client `evaluation/libero/libero_utils.resize_image` (TF JPEG round trip + Lanczos resize), the client's
    `cv2.imencode` JPEG upload (`evaluation/libero/vla_policy.py`, which writes the RGB array as if it were BGR), then
    the server's `resize_image` + `scale_and_resize` in deploy.py. Keep in sync with those three files.
    """
    import cv2
    import io
    import math

    import tensorflow as tf

    size = image.shape[:2]
    img = tf.io.decode_image(tf.image.encode_jpeg(image), expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy()
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok, "JPEG encode failed"
    pil = Image.open(io.BytesIO(encoded.tobytes()))

    w, h = pil.size  # deploy.resize_image
    left = min(max((w - h) // 2, 0), w - h)
    pil = pil.crop((left, 0, left + h, h)).resize((224, 224), resample=Image.LANCZOS)
    w, h = pil.size  # deploy.scale_and_resize(scale=0.9, centered)
    new_w, new_h = int(w * math.sqrt(0.9)), int(h * math.sqrt(0.9))
    mw, mh = int((w - new_w) * 0.5), int((h - new_h) * 0.5)
    return pil.crop((mw, mh, mw + new_w, mh + new_h)).resize((224, 224), resample=Image.LANCZOS)


def build_trajectories(data_root_dir, dataset, dataset_statistics, future_action_window_size, shard_id, num_shards,
                       resize_size=(224, 224)):
    """The training RLDS pipeline (normalization, action chunking, resizing), in file order and without augmentation."""
    from vla.datasets.rlds.dataset import apply_frame_transforms, apply_trajectory_transforms, make_dataset_from_rlds
    from vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
    from vla.datasets.rlds.utils.data_utils import NormalizationType

    per_dataset_kwargs, _ = get_oxe_dataset_kwargs_and_weights(
        Path(data_root_dir), [(dataset, 1.0)], load_camera_views=("primary",), load_depth=False, load_proprio=False,
        load_language=True, action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
    )
    kwargs = dict(per_dataset_kwargs[0])
    kwargs.pop("dataset_frame_transform_kwargs", None)
    trajectories, stats = make_dataset_from_rlds(
        **kwargs, train=True, shuffle=False, dataset_statistics=dataset_statistics, load_all_data_for_training=True,
    )
    if num_shards > 1:
        trajectories = trajectories.shard(num_shards, shard_id)
    trajectories = apply_trajectory_transforms(
        trajectories, train=False, window_size=1, future_action_window_size=future_action_window_size,
        skip_unlabeled=True, dataset_statistics=stats,
    )
    return apply_frame_transforms(trajectories, train=False, resize_size=resize_size or {})


def load_policy(ckpt: str, lock_path: Path):
    """Load MemoryVLA as deploy.py does (training config merged in, bf16); loads are serialized across shards."""
    import fcntl

    from vla import load_vla

    cfg, _ = read_run_config(ckpt)
    kwargs = {k: v for k, v in cfg.items() if k not in LOAD_KEYS_SKIPPED}
    kwargs["use_bf16"] = True
    with open(lock_path, "w") as lock:
        # Loading reads the full fp32 checkpoint into host RAM before moving it to the GPU
        fcntl.flock(lock, fcntl.LOCK_EX)
        vla = load_vla(model_id_or_path=ckpt, load_for_training=False, **kwargs)
        vla = vla.to("cuda").eval().to(torch.bfloat16)
    return vla


def prompt_ids(vla, instruction: str) -> torch.Tensor:
    """Same prompt as `MemoryVLA.predict_action`, closed by the tokens (29871, 2) whose EOS state is the cog token."""
    from transformers import LlamaTokenizerFast

    tokenizer = vla.vlm.llm_backbone.tokenizer
    assert isinstance(tokenizer, LlamaTokenizerFast), f"Unsupported tokenizer {type(tokenizer)}"
    builder = vla.vlm.get_prompt_builder()
    builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
    ids = tokenizer(builder.get_prompt(), truncation=True, return_tensors="pt").input_ids
    return torch.cat([ids, torch.tensor([[29871, 2]])], dim=1).to("cuda")


@torch.inference_mode()
def extract_tokens(vla, input_ids: torch.Tensor, images):
    """Raw (pre-memory) cog [b, D_cog] and per [b, N, D_per] tokens for a batch of frames (arrays or PIL images)."""
    image_transform = vla.vlm.vision_backbone.image_transform
    pixels = [image_transform(image if isinstance(image, Image.Image) else Image.fromarray(image)) for image in images]
    if isinstance(pixels[0], dict):
        pixel_values = {k: torch.stack([p[k] for p in pixels]).to("cuda", torch.bfloat16) for k in pixels[0]}
    else:
        pixel_values = torch.stack(pixels).to("cuda", torch.bfloat16)
    ids = input_ids.expand(len(images), -1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla.vlm(
            input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixel_values,
            output_hidden_states=True, return_dict=True,
        )
        cog = output.hidden_states[-1][:, -1, :]
        per = vla.per_compr(vla.vlm.vision_feats)
    return cog.half().cpu(), per.half().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help=".../<run>/checkpoints/<name>.pt")
    parser.add_argument("--data_root_dir", required=True)
    parser.add_argument("--dataset", default="libero_goal_no_noops")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_episodes", type=int, default=0, help="0 = all (debugging aid)")
    parser.add_argument("--frame_stride", type=int, default=8,
                        help="Keep every N-th frame: the actions executed per policy call (LIBERO evaluation: 8)")
    parser.add_argument("--deploy_view", choices=("none", "libero"), default="libero",
                        help="`libero`: preprocess frames exactly as the LIBERO evaluation does (libero_deploy_view)")
    args = parser.parse_args()

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")  # TensorFlow only feeds data here

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, run_dir = read_run_config(args.ckpt)
    with open(run_dir / "dataset_statistics.json", "r") as f:
        dataset_statistics = json.load(f)[args.dataset]
    future_action_window_size = cfg.get("future_action_window_size", 15)

    index_path = out_dir / f"index_{args.shard_id:02d}.jsonl"
    indexed = set()
    if index_path.exists():
        with open(index_path, "r") as f:
            indexed = {json.loads(line)["episode"] for line in f if line.strip()}

    trajectories = build_trajectories(
        args.data_root_dir, args.dataset, dataset_statistics, future_action_window_size, args.shard_id, args.num_shards,
        resize_size=None if args.deploy_view != "none" else (224, 224),  # the deployment view resizes by itself
    )
    vla = load_policy(args.ckpt, out_dir / ".load.lock")

    started, done, frames = time.time(), 0, 0
    with open(index_path, "a") as index:
        for k, traj in enumerate(trajectories.as_numpy_iterator()):
            if args.max_episodes and k >= args.max_episodes:
                break
            episode = f"s{args.shard_id:02d}_{k:06d}"
            base = out_dir / f"ep_{episode}"
            if episode in indexed:
                continue
            if not base.with_suffix(".pt").exists():
                kept = np.arange(0, len(traj["action"]), args.frame_stride)
                images = traj["observation"]["image_primary"][kept, -1]  # [T, H, W, 3]
                if args.deploy_view == "libero":
                    images = [libero_deploy_view(image) for image in images]
                instruction = traj["task"]["language_instruction"][0].decode()
                ids = prompt_ids(vla, instruction)
                cogs, pers = [], []
                for start in range(0, len(images), args.batch_size):
                    cog, per = extract_tokens(vla, ids, images[start : start + args.batch_size])
                    cogs.append(cog)
                    pers.append(per)
                env_steps = np.asarray(traj["observation"]["timestep"])[kept, -1].astype(np.int64)
                meta = {
                    "episode": episode, "file": base.name, "T": int(len(images)), "instruction": instruction,
                    "frame_stride": args.frame_stride, "deploy_view": args.deploy_view, "prompt": instruction,
                }
                np.save(f"{base}.per.npy", torch.cat(pers).numpy())
                torch.save(
                    {
                        "cog": torch.cat(cogs),
                        "actions": torch.from_numpy(np.asarray(traj["action"], dtype=np.float32)[kept]),
                        "timesteps": torch.arange(len(kept)),
                        "env_steps": torch.from_numpy(env_steps),
                        "meta": meta,
                    },
                    f"{base}.pt",  # written last: its presence marks a finished episode
                )
            else:
                meta = torch.load(f"{base}.pt", map_location="cpu", weights_only=False)["meta"]
            index.write(json.dumps(meta) + "\n")
            index.flush()
            done, frames = done + 1, frames + meta["T"]
            if done % 20 == 0:
                rate = frames / (time.time() - started)
                print(f"[shard {args.shard_id}] episodes={done} frames={frames} ({rate:.1f} frames/s)", flush=True)
    print(f"[shard {args.shard_id}] finished: episodes={done} frames={frames}", flush=True)


if __name__ == "__main__":
    main()
