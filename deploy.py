import json
import numpy as np
from PIL import Image
from typing import Optional
import os
import argparse
import yaml
from argparse import Namespace
import math
from flask import Flask, request, jsonify
import tempfile

import torch

from vla import load_vla
from vla.experience import ExperienceStore
from evaluation.simpler_env.adaptive_ensemble import AdaptiveEnsembler

app = Flask(__name__)


class MemVLAService:
    def __init__(
        self,
        saved_model_path: str = "",
        unnorm_key: str = None,
        image_size: list[int] = [224, 224],
        cfg_scale: float = 1.5,
        num_ddim_steps: int = 10, 
        use_ddim: bool = True,
        use_bf16: bool = False,
        action_ensemble: bool = True,
        adaptive_ensemble_alpha: float = 0.1,
        action_ensemble_horizon: int = 2,
        action_chunking: bool = False,
        action_chunking_window: Optional[int] = None,
        timestep_stride: int = 1,
        exp_store_dir: Optional[str] = None,
        exp_keyframes: int = 8,
        exp_keyframe_strategy: str = "uniform",
        exp_pin_timestep: str = "orig",
        exp_pin_target: str = "both",
        preserve_unmasked_actions: bool = False,
        exp_diag: bool = False,
        exp_diag_path: Optional[str] = None,
        args=None,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        assert not (action_chunking and action_ensemble), "Now 'action_chunking' and 'action_ensemble' cannot both be True."

        self.unnorm_key = unnorm_key

        print(f"*** unnorm_key: {unnorm_key} ***")

        kwargs = vars(args).copy()
        for k in [
            "model_id_or_path", "saved_model_path", "pretrained_checkpoint", "preserve_unmasked_actions",
        ]:
            kwargs.pop(k, None)

        self.vla = load_vla(
          model_id_or_path=saved_model_path,
          load_for_training=False,
          **kwargs,
        )
        self.vla = self.vla.to("cuda").eval()
        if use_bf16:
            print("Using bfloat16 inference mode (auto-conversion for all modules).")
            self.vla = self.vla.to(torch.bfloat16)
        else:
            print("Using standard float32 inference mode.")
            self.vla = self.vla.to(torch.float32)

        self.cfg_scale = cfg_scale

        self.image_size = image_size
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.action_chunking = action_chunking
        self.action_chunking_window = action_chunking_window
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None

        # Cross-episode experience (only active when the client uses /start_episode and /end_episode)
        self.timestep_stride = timestep_stride
        self.exp_store = ExperienceStore(exp_store_dir) if exp_store_dir else None
        self.exp_keyframes = exp_keyframes
        self.exp_keyframe_strategy = exp_keyframe_strategy
        self.exp_pin_timestep = exp_pin_timestep
        self.exp_pin_target = exp_pin_target
        self.preserve_unmasked_actions = preserve_unmasked_actions
        self.exp_diag = exp_diag
        self.exp_diag_path = exp_diag_path
        self.exp_diag_step = 0
        self.last_exp_diagnostics = {}
        self.vla.exp_diag_enabled = exp_diag
        for bank in (self.vla.cog_mem_bank, self.vla.per_mem_bank):
            bank.exp_diag_enabled = exp_diag
            if bank.fusion_type == "gate":
                bank.gate_fusion_blocks.collect_diag = exp_diag
        if exp_diag_path:
            os.makedirs(os.path.dirname(os.path.abspath(exp_diag_path)), exist_ok=True)
        self.episode = None

        self.args = args
        self.reset()

    def reset(self) -> None:
        if self.action_ensemble:
            self.action_ensembler.reset()

    def start_episode(self, meta: dict) -> dict:
        """
        meta: suite, task_id, init_id, trial, seed, exp_mode, exp_k, record
        Resets memory, seeds the diffusion noise, pins retrieved experiences and (optionally) starts recording.
        """
        self.reset()
        self.vla.cog_mem_bank.reset()
        self.vla.per_mem_bank.reset()
        self.vla.cur_timestep = 0
        self.exp_diag_step = 0

        # Same seed => same diffusion noise sequence, so conditions can be compared pairwise
        seed = int(meta.get("seed", 0))
        torch.manual_seed(seed)

        exp_mode = meta.get("exp_mode", "none")
        metas, cog_entries, per_entries = [], [], []
        if exp_mode != "none":
            assert self.exp_store is not None, "Start the server with `--exp_store_dir` to use experiences"
            metas = self.exp_store.query(
                exp_mode, meta["suite"], int(meta["task_id"]), int(meta["init_id"]),
                int(meta.get("exp_k", 1)), seed, cur_meta=meta,
            )
            cog_entries, per_entries = self.exp_store.build_pinned_entries(
                metas,
                num_keyframes=self.exp_keyframes,
                strategy=self.exp_keyframe_strategy,
                timestep_mode=self.exp_pin_timestep,
                target=self.exp_pin_target,
                device=torch.device("cuda"),
                cog_dtype=next(self.vla.cog_mem_bank.parameters()).dtype,
                per_dtype=next(self.vla.per_mem_bank.parameters()).dtype,
                seed=seed,
            )
        self.vla.cog_mem_bank.set_pinned(cog_entries)
        self.vla.per_mem_bank.set_pinned(per_entries)

        record = bool(meta.get("record", False))
        assert not record or self.exp_store is not None, "Start the server with `--exp_store_dir` to record experiences"
        self.episode = {"meta": dict(meta), "record": record, "cog": [], "per": [], "actions": [], "timesteps": []}

        return {
            "experiences": [m["name"] for m in metas],
            "num_pinned_cog": len(cog_entries),
            "num_pinned_per": len(per_entries),
        }

    def end_episode(self, success: bool, info: dict) -> dict:
        episode, self.episode = self.episode, None
        self.vla.cog_mem_bank.clear_pinned()
        self.vla.per_mem_bank.clear_pinned()

        if episode is None or not episode["record"] or len(episode["cog"]) == 0:
            return {"saved": None}

        meta = {**episode["meta"], **info, "success": bool(success), "num_calls": len(episode["cog"])}
        path = self.exp_store.save(
            meta,
            cog=torch.stack(episode["cog"]),
            per=torch.stack(episode["per"]),
            actions=np.stack(episode["actions"]),
            timesteps=np.asarray(episode["timesteps"]),
        )
        return {"saved": path}

    def step(
        self,
        image: str,
        task_description: str = None,
        episode_first_frame: str = 'False',
        *args, **kwargs,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: Path to the image file
            task_description: Optional[str], task description
            episode_first_frame: 'True' or 'False', whether the current frame is the first frame of an episode
        Output:
            action: list[float], the ensembled 7-DoFs action of End-effector and gripper

        """

        assert episode_first_frame in ['True', 'False']

        if episode_first_frame == 'True':
            self.reset()

        image: Image.Image = Image.open(image)

        # [IMPORTANT!]: Please process the input images here in exactly the same way as the images
        # were processed during finetuning to ensure alignment between inference and training.
        # Make sure, as much as possible, that the gripper is visible in the processed images.
        resized_image = resize_image(image, size=self.image_size)

        # save resized image for debugging
        resized_image.save("resized_image.png")
        recording = self.episode is not None and self.episode["record"]
        self.last_exp_diagnostics = {}
        outputs = self.vla.predict_action(
            image=resized_image,
            instruction=task_description,
            unnorm_key=self.unnorm_key,
            cfg_scale=self.cfg_scale,
            use_ddim=self.use_ddim,
            num_ddim_steps=self.num_ddim_steps,
            episode_first_frame=episode_first_frame,
            timestep_stride=self.timestep_stride,
            return_features=recording,
            preserve_unmasked_actions=self.preserve_unmasked_actions,
        )
        if recording:
            unnormed_actions, normalized_actions, features = outputs
            self.episode["cog"].append(features["cog"].half().cpu())
            self.episode["per"].append(features["per"].half().cpu())
            self.episode["actions"].append(normalized_actions.copy())
            self.episode["timesteps"].append(self.vla.cur_timestep - self.timestep_stride)
        else:
            unnormed_actions, normalized_actions = outputs

        if self.exp_diag and self.exp_diag_path:
            self.last_exp_diagnostics = getattr(self.vla, "last_exp_diagnostics", {})
            meta = (self.episode or {}).get("meta", {})
            row = {
                "suite": meta.get("suite"), "task_id": meta.get("task_id"),
                "init_id": meta.get("init_id"), "trial": meta.get("trial"),
                "exp_mode": meta.get("exp_mode"), "step": self.exp_diag_step,
                **self.last_exp_diagnostics,
            }
            with open(self.exp_diag_path, "a", encoding="utf-8") as diag_file:
                diag_file.write(json.dumps(row, allow_nan=False) + "\n")
            self.exp_diag_step += 1

        if self.action_ensemble:
            unnormed_actions = self.action_ensembler.ensemble_action(unnormed_actions)
            # Translate the value of the gripper's open/close state to 0 or 1.
            # Please adjust this line according to the control mode of different grippers.
            unnormed_actions[6] = unnormed_actions[6] > 0.5
            action = unnormed_actions.tolist()
        elif self.action_chunking:
            # [IMPORTANT!]: Please modify the code here to output multiple actions at once.
            # The code below only outputs the first action in the chunking.
            # The chunking window size can be adjusted by modifying the 'action_chunking_window' parameter.
            if self.action_chunking_window is not None:
                chunked_actions = []
                for i in range(0, self.action_chunking_window):
                    chunked_actions.append(unnormed_actions[i].tolist())
                action = chunked_actions
            else:
                raise ValueError("Please specify the 'action_chunking_window' when using action chunking.")
        else:
            # Output the first action in the chunking. Can be modified to output multiple actions at once.
            unnormed_actions = unnormed_actions[0]
            action = unnormed_actions.tolist()

        print(f"Instruction: {task_description}")
        # print(f"Model path: {self.args.saved_model_path} at port {self.args.port}")
        return action


# [IMPORTANT!]: Please modify the image processing code here to ensure that the input images  
# are handled in exactly the same way as during the finetuning phase.
# Make sure, as much as possible, that the gripper is visible in the processed images.
def resize_image(image: Image, size=(224, 224), shift_to_left=0):
    w, h = image.size
    #assert h < w, "Height should be less than width"
    left_margin = (w - h) // 2 - shift_to_left
    left_margin = min(max(left_margin, 0), w - h)
    image = image.crop((left_margin, 0, left_margin + h, h))

    image = image.resize(size, resample=Image.LANCZOS)
    
    image = scale_and_resize(image, target_size=(224, 224), scale=0.9, margin_w_ratio=0.5, margin_h_ratio=0.5)
    return image

# Here the image is first center cropped and then resized back to its original size 
# because random crop data augmentation was used during finetuning.
def scale_and_resize(image : Image, target_size=(224, 224), scale=0.9, margin_w_ratio=0.5, margin_h_ratio=0.5):
    w, h = image.size
    new_w = int(w * math.sqrt(scale))
    new_h = int(h * math.sqrt(scale))
    margin_w_max = w - new_w
    margin_h_max = h - new_h
    margin_w = int(margin_w_max * margin_w_ratio)
    margin_h = int(margin_h_max * margin_h_ratio)
    image = image.crop((margin_w, margin_h, margin_w + new_w, margin_h + new_h))
    image = image.resize(target_size, resample=Image.LANCZOS)
    return image


parser = argparse.ArgumentParser()
parser.add_argument("--saved_model_path", type=str, default="")
parser.add_argument("--unnorm_key", type=str, default='custom_finetuning')
parser.add_argument("--image_size", type=list[int], default=[224, 224])
parser.add_argument("--cfg_scale", type=float, default=1.5)
parser.add_argument("--port", type=int, default=2345)
parser.add_argument("--use_bf16", action="store_true")
parser.add_argument("--action_ensemble", action="store_true")
parser.add_argument("--action_ensemble_horizon", type=int, default=2)
parser.add_argument("--adaptive_ensemble_alpha", type=float, default=0.1)
parser.add_argument("--action_chunking", action="store_true")
parser.add_argument("--action_chunking_window", type=int, default=None)
parser.add_argument("--timestep_stride", type=int, default=1)
parser.add_argument("--exp_store_dir", type=str, default=None)
parser.add_argument("--exp_keyframes", type=int, default=8)
parser.add_argument("--exp_keyframe_strategy", type=str, default="uniform", choices=["uniform", "event"])
parser.add_argument("--exp_pin_timestep", type=str, default="orig", choices=["orig", "neg", "zero"])
parser.add_argument("--exp_pin_target", type=str, default="both", choices=["both", "cog", "per"])
parser.add_argument("--preserve_unmasked_actions", action="store_true")
parser.add_argument("--exp_diag", action="store_true")
parser.add_argument("--exp_diag_path", type=str, default=None)

args = parser.parse_args()

with open(os.path.join(os.path.dirname(os.path.dirname(args.saved_model_path)), "config.yaml"), "r") as f:
    yaml_args = yaml.safe_load(f) or {}

def deep_update(base: dict, updates: dict):
    """Recursively merge two dictionaries, with updates taking precedence but preserving keys from base."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

cli_args = vars(args)
merged_args = deep_update(yaml_args.copy(), cli_args)

args = Namespace(**merged_args)

inferencer = MemVLAService(
    saved_model_path=args.saved_model_path,
    unnorm_key=args.unnorm_key,
    image_size=args.image_size,
    cfg_scale=args.cfg_scale,
    use_bf16=args.use_bf16,
    action_ensemble=args.action_ensemble,
    adaptive_ensemble_alpha=args.adaptive_ensemble_alpha,
    action_ensemble_horizon=args.action_ensemble_horizon,
    action_chunking=args.action_chunking,
    action_chunking_window=args.action_chunking_window,
    timestep_stride=args.timestep_stride,
    exp_store_dir=args.exp_store_dir,
    exp_keyframes=args.exp_keyframes,
    exp_keyframe_strategy=args.exp_keyframe_strategy,
    exp_pin_timestep=args.exp_pin_timestep,
    exp_pin_target=args.exp_pin_target,
    preserve_unmasked_actions=args.preserve_unmasked_actions,
    exp_diag=args.exp_diag,
    exp_diag_path=args.exp_diag_path,
    args=args,
)


@app.route('/process_frame', methods=['POST'])
def inference():
    # Check if image is provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image = request.files['image']

    # Check if text is provided
    if 'text' not in request.form:
        return jsonify({'error': 'No text provided'}), 400
    query = request.form['text']

    # Check if episode_first_frame is provided
    if 'episode_first_frame' not in request.form:
        return jsonify({'error': 'No episode_first_frame provided'}), 400
    episode_first_frame = request.form['episode_first_frame']

    # Save image to temporary file and resize to expected dimensions
    with tempfile.NamedTemporaryFile(delete=False) as temp_image:
        image.save(temp_image.name)
        temp_image_path = temp_image.name

    # Construct input query and prepare for inference
    input_query = {
        'task_description': query,
        'episode_first_frame': episode_first_frame,
    }

    # Run inference
    answer = inferencer.step(temp_image_path, **input_query)
    print(answer)

    # Convert action array to string based on different modes
    if inferencer.action_ensemble:
        # For action ensemble mode, directly convert the action list
        action_str = ' '.join([str(x) for x in answer])
    elif inferencer.action_chunking:
        # For action chunking mode, convert the chunked actions
        action_str = ';'.join([' '.join([str(x) for x in chunk]) for chunk in answer])
    else:
        # For single action mode
        action_str = ' '.join([str(x) for x in answer])

    return jsonify({'response': action_str})


@app.route('/start_episode', methods=['POST'])
def start_episode():
    return jsonify(inferencer.start_episode(request.get_json(force=True)))


@app.route('/end_episode', methods=['POST'])
def end_episode():
    payload = request.get_json(force=True)
    return jsonify(inferencer.end_episode(bool(payload.get('success', False)), payload.get('info') or {}))


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=args.port)
