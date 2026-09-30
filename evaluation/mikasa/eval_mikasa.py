"""Paired cross-episode experience evaluation on the original MIKASA-Robo tasks."""

import argparse
import hashlib
import json
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import gymnasium as gym
import mikasa_robo_suite  # noqa: F401 - registers the benchmark environments
import numpy as np
import requests
import torch
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mikasa_robo_suite.utils.wrappers import (
    DebugRewardWrapper,
    InitialZeroActionWrapper,
    RememberColorInfoWrapper,
    RenderRewardInfoWrapper,
    RenderStepInfoWrapper,
    StateOnlyTensorToDictWrapper,
)
from PIL import Image

from evaluation.libero.vla_policy import LLaVAClient


# Task order and prompts match the evaluation log published with memvla-mikasa.pt.
TASKS = {
    0: ("SG", "ShellGameTouch-v0", "Memorize the position of the cup covering the ball, then pick that cup", 90),
    1: ("IM", "InterceptMedium-v0", "Track the ball\u2019s movement, estimate its velocity, then aim the ball at the target", 90),
    2: ("RC3", "RememberColor3-v0", "Remember the color of the cube and then pick the matching one", 60),
    3: ("RC5", "RememberColor5-v0", "Remember the color of the cube and then pick the matching one", 60),
    4: ("RC9", "RememberColor9-v0", "Remember the color of the cube and then pick the matching one", 60),
}
EXP_MODES = (
    "none", "other_init_success", "same_init_success", "same_init_failure",
    "other_task_success", "noise", "other_init_success_same_color", "other_init_success_diff_color",
)
ACTION_DIM = 7


class MikasaClient(LLaVAClient):
    def process_frame(self, text: str, episode_first_frame: str, base_cam: np.ndarray) -> str:
        buffer = BytesIO()
        Image.fromarray(base_cam).save(buffer, format="PNG")
        response = requests.post(
            self.base_url + "/process_frame",
            data={"text": text, "episode_first_frame": episode_first_frame},
            files={"image": ("frame.png", buffer.getvalue(), "image/png")},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]


def parse_action_chunk(response: str) -> np.ndarray:
    if not response:
        raise ValueError("Policy returned no actions")
    chunks = [np.fromstring(part, sep=" ", dtype=np.float32) for part in response.split(";")]
    if any(chunk.size != ACTION_DIM for chunk in chunks):
        raise ValueError(f"Expected 7 values per action; got {[chunk.size for chunk in chunks]}")
    return np.stack(chunks)


def camera_image(obs: dict) -> np.ndarray:
    return obs["sensor_data"]["base_camera"]["rgb"][0].detach().cpu().numpy()


def flag(value) -> bool:
    return bool(torch.as_tensor(value).reshape(-1)[0].item())


def make_env(env_id: str) -> gym.Env:
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        render_mode="all",
        reconfiguration_freq=1,
        sim_backend="gpu",
        sensor_configs={"width": 128, "height": 128, "shader_pack": "default"},
    )
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)
    env = StateOnlyTensorToDictWrapper(env)
    env = InitialZeroActionWrapper(env, n_initial_steps=0)
    if env_id.startswith("RememberColor"):
        env = RememberColorInfoWrapper(env)
    env = RenderStepInfoWrapper(env)
    env = RenderRewardInfoWrapper(env)
    return DebugRewardWrapper(env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, choices=TASKS.keys(), required=True)
    parser.add_argument("--num_inits", type=int, default=10)
    parser.add_argument("--trials_per_init", type=int, default=1)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--exp_mode", choices=EXP_MODES, default="none")
    parser.add_argument("--exp_k", type=int, default=1)
    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--record_experience", action="store_true")
    parser.add_argument("--run_id_note", default="")
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=2345)
    args = parser.parse_args()

    if args.record_experience and args.exp_mode != "none":
        parser.error("Experience collection requires --exp_mode none")

    task, env_id, instruction, max_steps = TASKS[args.task_id]
    args.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    results_path = args.log_dir / f"mikasa-task{args.task_id}-{args.run_id_note}-{stamp}_results.jsonl"
    policy = MikasaClient(base_url=f"http://127.0.0.1:{args.port}")
    env = make_env(env_id)

    try:
        with results_path.open("a", buffering=1) as results:
            for init_id in range(args.num_inits):
                for trial in range(args.trials_per_init):
                    started = time.perf_counter()
                    # Reusing the reset seed gives the same scene for every trial and mode.
                    env_seed = init_id + 1
                    obs, _ = env.reset(seed=env_seed, options={})
                    target_color = None
                    if task in ("RC3", "RC5", "RC9"):
                        target_color = int(torch.as_tensor(obs["oracle_info"]).reshape(-1)[0].item())
                    initial_image_hash = hashlib.sha256(camera_image(obs).tobytes()).hexdigest()[:16]
                    episode_seed = args.seed_offset + args.task_id * 100000 + init_id * 100 + trial
                    exp_info = policy.start_episode(
                        suite="mikasa", task_id=args.task_id, init_id=init_id, trial=trial, seed=episode_seed,
                        exp_mode=args.exp_mode, exp_k=args.exp_k, record=args.record_experience,
                        target_color=target_color,
                    )
                    print(f"{task} init={init_id} trial={trial} Pinned experiences: {exp_info}", flush=True)

                    success, success_once, steps, first_frame = False, False, 0, "True"
                    finished = False
                    while steps < max_steps:
                        response = policy.process_frame(
                            text=instruction, episode_first_frame=first_frame, base_cam=camera_image(obs)
                        )
                        first_frame = "False"
                        for action in parse_action_chunk(response):
                            action[:3] *= args.action_scale
                            obs, _, _, _, info = env.step(action)
                            steps += 1
                            success_once = success_once or flag(info.get("success", False))
                            if "final_info" in info:
                                episode = info["final_info"].get("episode", {})
                                success = flag(episode.get("success_once", success_once))
                                finished = True
                                break
                            if steps >= max_steps:
                                break
                        if finished:
                            break
                    if not finished:
                        success = success_once

                    policy.end_episode(success, env_steps=steps, target_color=target_color, env_seed=env_seed)
                    row = {
                        "suite": "mikasa", "task_id": args.task_id, "task": task, "env_id": env_id,
                        "init_id": init_id, "trial": trial, "seed": episode_seed, "seed_offset": args.seed_offset,
                        "exp_mode": args.exp_mode, "exp_k": args.exp_k, "record": args.record_experience,
                        "experiences": exp_info["experiences"],
                        "num_pinned_cog": exp_info["num_pinned_cog"],
                        "num_pinned_per": exp_info["num_pinned_per"],
                        "target_color": target_color,
                        "success": success, "env_steps": steps, "env_seed": env_seed,
                        "initial_image_hash": initial_image_hash,
                        "duration_s": round(time.perf_counter() - started, 2),
                    }
                    results.write(json.dumps(row) + "\n")
                    print(f"{task} init={init_id} trial={trial}: success={success} steps={steps} "
                          f"duration={row['duration_s']}s", flush=True)
    finally:
        env.close()

    print(f"Results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
