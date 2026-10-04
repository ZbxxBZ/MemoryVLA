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
import threading
import time

import torch

from vla import load_vla
from vla.experience_retrieval import ExperienceRuntime, ExperienceStore, load_adapter, normalize_instruction
from evaluation.simpler_env.adaptive_ensemble import AdaptiveEnsembler

app = Flask(__name__)

# Server-only options, not forwarded to the model constructor
SERVER_ONLY_KEYS = (
    "model_id_or_path", "saved_model_path", "pretrained_checkpoint", "exp_adapter", "exp_library", "exp_num_refs",
    "exp_window", "exp_select", "exp_record_dir",
)


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
        exp_adapter: Optional[str] = None,
        exp_library: Optional[list] = None,
        exp_num_refs: int = 4,
        exp_window: Optional[int] = None,
        exp_select: str = "similar",
        exp_record_dir: Optional[str] = None,
        args=None,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        assert not (action_chunking and action_ensemble), "Now 'action_chunking' and 'action_ensemble' cannot both be True."

        self.unnorm_key = unnorm_key

        print(f"*** unnorm_key: {unnorm_key} ***")

        kwargs = vars(args).copy()
        for k in SERVER_ONLY_KEYS:
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

        # Cross-episode experience retrieval (vla/experience_retrieval.py); only active for clients that use
        # /start_episode and /end_episode. Without an adapter, episodes can still be recorded into stores.
        self.executed_per_call = action_chunking_window if action_chunking else 1
        self.exp_num_refs, self.exp_select = exp_num_refs, exp_select
        self.exp_window, self.exp_max_refs = exp_window or 3, 8
        self.exp_chunk_shape = (self.vla.future_action_window_size + 1, self.vla.action_model.in_channels)
        self.has_adapter = False
        if exp_adapter:
            adapter, ckpt = load_adapter(exp_adapter)
            dtype = next(self.vla.action_model.parameters()).dtype
            self.vla.attach_exp_adapter(adapter.to("cuda", dtype).eval())
            self.has_adapter = True
            self.exp_window = exp_window or ckpt.get("window", 3)
            self.exp_max_refs = adapter.config["max_refs"]
            assert self.exp_chunk_shape == (adapter.config["chunk_len"], adapter.config["action_dim"]), \
                f"Adapter chunk {adapter.config['chunk_len']}x{adapter.config['action_dim']} != model {self.exp_chunk_shape}"
            print(f"Experience adapter {exp_adapter} (step {ckpt.get('step')}): window {self.exp_window}, "
                  f"gates {adapter.gates()}")
        self.library = ExperienceStore.load(exp_library) if exp_library else ExperienceStore()
        if exp_library:
            print(f"Experience library: {len(self.library)} episodes, {len(self.library.by_task)} tasks")
        self.stores = {}  # store name -> ExperienceStore (the library plus the rollouts recorded into it)
        self.exp_record_dir = exp_record_dir
        if exp_record_dir:
            os.makedirs(exp_record_dir, exist_ok=True)

        # Sessions: several simulator clients can share this server. Each session owns its memory banks, timestep,
        # current episode and RNG stream; they are swapped in under a lock. Requests without `session` use "default".
        self.lock = threading.RLock()
        self.sessions = {}
        self.active_session = "default"
        self.episode = None  # {"runtime", "store", "store_name", "record", "meta"} of the active session

        self.args = args
        self.reset()

    def reset(self) -> None:
        if self.action_ensemble:
            self.action_ensembler.reset()

    def _session_state(self) -> dict:
        return {
            "cog_bank": self.vla.cog_mem_bank.bank, "per_bank": self.vla.per_mem_bank.bank,
            "cur_timestep": self.vla.cur_timestep, "episode": self.episode,
            "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
        }

    def activate_session(self, session: str) -> None:
        if session == self.active_session:
            return
        self.sessions[self.active_session] = self._session_state()
        state = self.sessions.pop(session, None)
        if state is None:
            state = {
                "cog_bank": {}, "per_bank": {}, "cur_timestep": 0, "episode": None,
                "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
            }
        self.vla.cog_mem_bank.bank, self.vla.per_mem_bank.bank = state["cog_bank"], state["per_bank"]
        self.vla.cur_timestep, self.episode = state["cur_timestep"], state["episode"]
        self.vla.exp_provider = self.episode["runtime"] if self.episode else None
        torch.set_rng_state(state["rng_cpu"])
        torch.cuda.set_rng_state(state["rng_cuda"])
        self.active_session = session

    def get_store(self, name: str) -> ExperienceStore:
        if name not in self.stores:
            self.stores[name] = self.library.copy()
        return self.stores[name]

    def start_episode(self, meta: dict) -> dict:
        """
        meta: task_key (instruction), seed, condition (none | real | unaligned), store (name, default: the session),
        sources ("demo,rollout"), record (add the episode to the store at the end), num_refs, select, plus anything
        the client wants echoed into the recorded episode.
        Resets the episodic memory and seeds the diffusion noise, so conditions can be compared on paired seeds.
        """
        self.reset()
        self.vla.cog_mem_bank.reset()
        self.vla.per_mem_bank.reset()
        self.vla.cur_timestep = 0
        seed = int(meta.get("seed", 0))
        torch.manual_seed(seed)

        condition = meta.get("condition", "none")
        assert condition == "none" or self.has_adapter, "Start the server with --exp_adapter to read experiences"
        store_name = meta.get("store") or self.active_session
        store = self.get_store(store_name)
        task_key = normalize_instruction(meta["task_key"])
        sources = tuple(s for s in str(meta.get("sources", "demo,rollout")).split(",") if s)
        runtime = ExperienceRuntime(
            store, task_key, condition=condition, num_refs=int(meta.get("num_refs") or self.exp_num_refs),
            window=self.exp_window, select=meta.get("select") or self.exp_select, sources=sources,
            max_refs=self.exp_max_refs, chunk_shape=self.exp_chunk_shape, seed=seed, device="cuda",
        )
        self.episode = {"runtime": runtime, "store": store, "store_name": store_name,
                        "record": bool(meta.get("record", False)), "meta": dict(meta)}
        self.vla.exp_provider = runtime
        return {
            "task_key": task_key, "store": store_name, "store_size": len(store),
            "candidates": len(store.candidates(task_key, sources=sources)),
        }

    def end_episode(self, success: bool, info: dict) -> dict:
        episode, self.episode = self.episode, None
        self.vla.exp_provider = None
        if episode is None:
            return {}
        runtime = episode["runtime"]
        out = runtime.info()
        if episode["record"]:
            name = f"{episode['store_name']}/{time.time_ns()}"
            recorded = runtime.to_episode(success, self.executed_per_call, name=name,
                                          meta={**episode["meta"], **info, "success": bool(success)})
            if recorded is not None:
                episode["store"].add(recorded)
                out["added"] = name
                if self.exp_record_dir:
                    path = os.path.join(self.exp_record_dir, name.replace("/", "__") + ".pt")
                    torch.save({"version": 1, "episodes": [recorded]}, path)
        out["store_size"] = len(episode["store"])
        return out

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
        unnormed_actions, normalized_actions = self.vla.predict_action(
            image=resized_image,
            instruction=task_description,
            unnorm_key=self.unnorm_key,
            cfg_scale=self.cfg_scale,
            use_ddim=self.use_ddim,
            num_ddim_steps=self.num_ddim_steps,
            episode_first_frame=episode_first_frame,
        )
        if self.episode is not None:
            self.episode["runtime"].record_prediction(normalized_actions)

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
parser.add_argument("--exp_adapter", type=str, default=None, help="Trained experience adapter (train_experience.py)")
parser.add_argument("--exp_library", type=str, nargs="+", default=None,
                    help="Experience store file(s) every store starts from (build_library.py)")
parser.add_argument("--exp_num_refs", type=int, default=4, help="References per episode (clients may override)")
parser.add_argument("--exp_window", type=int, default=None, help="Pointer look-ahead; default: the adapter's")
parser.add_argument("--exp_select", choices=("similar", "recent", "random"), default="similar",
                    help="Reference choice among the task's successful episodes (clients may override)")
parser.add_argument("--exp_record_dir", type=str, default=None, help="Also save every recorded episode here")

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
    exp_adapter=args.exp_adapter,
    exp_library=args.exp_library,
    exp_num_refs=args.exp_num_refs,
    exp_window=args.exp_window,
    exp_select=args.exp_select,
    exp_record_dir=args.exp_record_dir,
    args=args,
)


@app.route('/process_frame', methods=['POST'])
def inference():
    session = request.form.get('session', 'default')
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
    with inferencer.lock:
        inferencer.activate_session(session)
        answer = inferencer.step(temp_image_path, **input_query)
    os.remove(temp_image_path)
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
    payload = request.get_json(force=True)
    with inferencer.lock:
        inferencer.activate_session(payload.pop('session', 'default'))
        return jsonify(inferencer.start_episode(payload))


@app.route('/end_episode', methods=['POST'])
def end_episode():
    payload = request.get_json(force=True)
    with inferencer.lock:
        inferencer.activate_session(payload.get('session', 'default'))
        return jsonify(inferencer.end_episode(bool(payload.get('success', False)), payload.get('info') or {}))


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=args.port)
