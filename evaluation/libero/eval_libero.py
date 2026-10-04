import json
import os
from dataclasses import dataclass
from typing import List, Optional, Union
import draccus
import numpy as np
import tqdm

os.environ['MUJOCO_GL'] = 'osmesa'
# apt install -y libosmesa6-dev libgl1-mesa-dev libglu1-mesa-dev

from libero.libero import benchmark

from libero_utils import (
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from robot_utils import DATE_TIME, set_seed_everywhere

# let tensorflow only see CPU
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


@dataclass
class GenerateConfig:
    # fmt: off
    task_suite_name: str = "libero_spatial" # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10 # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50 # Number of rollouts per task
    spcial_task_id: Union[List[int], int, None] = None # List of task IDs to evaluate (default: None, evaluates all tasks in the suite)
    run_id_note: str = ""
    local_log_dir: str = "./logs/eval_libero" # Local directory for eval logs
    seed: int = 7 # Random Seed (for reproducibility)
    resolution: Union[int, tuple] = 256 # Image resolution for model input
    port: int = 6800
    save_video: bool = True # Save a replay video per episode
    agentview_only: bool = False # Render only the camera the policy reads

    # Episodes: by default the first `num_trials_per_task` initial states, once each
    init_ids: Optional[List[int]] = None # Initial states to run
    trials_per_init: int = 1 # Rollouts per initial state
    episode_order: str = "init_major" # init_major: the trials of an init in a row | trial_major: every init once, then again
    seed_offset: int = 0 # Offset of the per-episode seed (diffusion noise on the server); same seed => paired episodes

    # Cross-episode experience (deploy.py /start_episode and /end_episode)
    episode_api: bool = False # Call /start_episode and /end_episode around every episode
    session_id: str = "" # Server session, so several clients can share one server; default: <run_id_note>-<pid>
    exp_condition: str = "none" # none | real | unaligned (control: random reference steps)
    exp_store: str = "" # Server-side experience store; default: the session. Starts as a copy of --exp_library
    exp_sources: str = "demo,rollout" # Episodes a reference may come from
    exp_record: bool = False # Add every episode of this run to the store (references only use the successful ones)
    exp_num_refs: int = 0 # References per episode; 0 = the server's --exp_num_refs
    exp_select: str = "" # similar | recent | random; "" = the server's --exp_select

    # LIBERO-plus: every task is a distinct perturbed scene. `group_file` is a JSON {suite: {"group": [...],
    # "category": [...], "difficulty": [...], "name": [...], "instruction": [...]}} indexed by task id; experiences
    # are then keyed by the base task (the instruction of the task's group) instead of the task's own description.
    group_file: Optional[str] = None
    # LIBERO-plus builds `task.language` from the file name, so outside the language dimension it ends with the
    # perturbation id ("... view 6 15 100 0 0 initstate 0"). With `meta_instruction` the group file's per-task
    # "instruction" is sent instead (null keeps `task.language`, e.g. for the rewritten language-perturbed instructions).
    meta_instruction: bool = False
    # fmt: on


def base_task_keys(task_meta):
    """LIBERO-plus group -> base instruction (from the group's tasks that carry an instruction)."""
    keys = {}
    for group, instruction in zip(task_meta["group"], task_meta.get("instruction") or [None] * len(task_meta["group"])):
        if instruction and group not in keys:
            keys[group] = instruction
    return keys


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    if cfg.spcial_task_id is not None and isinstance(cfg.spcial_task_id, int):
        cfg.spcial_task_id = [cfg.spcial_task_id]
    if cfg.init_ids is not None and isinstance(cfg.init_ids, int):
        cfg.init_ids = [cfg.init_ids]
    if cfg.exp_condition != "none" or cfg.exp_record:
        assert cfg.episode_api, "`exp_condition` / `exp_record` require `--episode_api True`"
    assert cfg.episode_order in ("init_major", "trial_major"), f"Unknown episode_order {cfg.episode_order}"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize local logging
    run_id = f"{cfg.task_suite_name}-{cfg.num_trials_per_task}trials-seed{cfg.seed}-{cfg.run_id_note}-{DATE_TIME}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    results_filepath = os.path.join(cfg.local_log_dir, run_id + "_results.jsonl")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = cfg.resolution # TODO need to be done in the cfg

    ################################################################
    ### import Policy
    from vla_policy import LLaVAClient
    session = cfg.session_id or f"{cfg.run_id_note}-{os.getpid()}"
    policy = LLaVAClient(base_url=f'http://localhost:{cfg.port}', session=session)
    task_meta, group_keys = None, {}
    if cfg.group_file:
        with open(cfg.group_file) as f:
            task_meta = json.load(f)[cfg.task_suite_name]
        group_keys = base_task_keys(task_meta)
    ################################################################

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Skip tasks if specified
        if cfg.spcial_task_id is not None and task_id not in cfg.spcial_task_id:
            print(f"Skipping task {task_id}...")
            continue

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, resolution=256, agentview_only=cfg.agentview_only)
        if cfg.meta_instruction:
            assert task_meta is not None and "instruction" in task_meta, "`meta_instruction` needs a group file with `instruction`"
            task_description = task_meta["instruction"][task_id] or task_description
        # Experiences are shared by episodes of the same base task
        if task_meta is not None:
            group = task_meta["group"][task_id]
            task_key = group_keys.get(group, f"{cfg.task_suite_name}/group{group}")
        else:
            task_key = task_description

        # Start episodes
        init_ids = cfg.init_ids if cfg.init_ids is not None else list(range(cfg.num_trials_per_task))
        if cfg.episode_order == "init_major":
            episode_list = [(init_id, trial) for init_id in init_ids for trial in range(cfg.trials_per_init)]
        else:
            episode_list = [(init_id, trial) for trial in range(cfg.trials_per_init) for init_id in init_ids]
        task_episodes, task_successes = 0, 0
        for init_id, trial in tqdm.tqdm(episode_list):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Same (task, init, trial) => same seed across conditions, so runs can be paired
            episode_seed = cfg.seed_offset + task_id * 100000 + init_id * 100 + trial

            # Reset environment
            env.reset()
            policy.reset()
            episode_first_frame = 'True'
            start_info = {}
            if cfg.episode_api:
                start_info = policy.start_episode(
                    task_key=task_key, seed=episode_seed, condition=cfg.exp_condition, store=cfg.exp_store,
                    sources=cfg.exp_sources, record=cfg.exp_record, num_refs=cfg.exp_num_refs, select=cfg.exp_select,
                    suite=cfg.task_suite_name, task_id=task_id, init_id=init_id, trial=trial,
                )
                print(f"Experience: {start_info}")
                log_file.write(f"Experience: {start_info}\n")

            # Set initial states
            obs = env.set_init_state(initial_states[init_id])
            done = False

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    observation = {
                        "base_cam": img,
                        "states": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    ###############################################################################
                    ### here we need to use flask to get the action from the model
                    ### we need a policy to get the action
                    action = policy.process_frame(text=task_description,
                                                  episode_first_frame=episode_first_frame,
                                                  **observation)

                    if ';' in action:
                        action = action.replace(';', ' ')

                    # str to np array
                    action = action.split(' ')
                    action = [float(x) for x in action]
                    action = np.array(action, dtype=float)

                    episode_first_frame = 'False'
                    action_dim = 7

                    # Adjust gripper action values according to your data's gripper definition
                    for i in range(len(action)):
                        if i % action_dim == action_dim - 1:
                            if action[i] == 1.0:
                                action[i] = -1.0
                            elif action[i] == 0.0:
                                action[i] = 1.0

                    done_flag = False
                    chunk_size = len(action) // action_dim
                    for i in range(chunk_size):
                        action_chunk = action[i * action_dim:(i + 1) * action_dim]

                        # Execute action in environment
                        obs, reward, done, info = env.step(action_chunk)
                        if done:
                            task_successes += 1
                            total_successes += 1
                            done_flag = True
                            break
                        t += 1

                    if done_flag:
                        break

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            extra = {}
            if task_meta:
                extra = {k: task_meta[k][task_id] for k in ("group", "category", "difficulty", "name") if k in task_meta}
            if cfg.episode_api:
                end_info = policy.end_episode(done, env_steps=t)
                extra["exp_info"] = {**start_info, **end_info}
            with open(results_filepath, "a") as f:
                f.write(json.dumps({
                    "suite": cfg.task_suite_name, "task_id": task_id, "task": task_description, "task_key": task_key,
                    "init_id": init_id, "trial": trial, "seed": episode_seed, "seed_offset": cfg.seed_offset,
                    "exp_condition": cfg.exp_condition, "exp_store": cfg.exp_store or session,
                    "exp_sources": cfg.exp_sources, "exp_record": cfg.exp_record,
                    "meta_instruction": cfg.meta_instruction, "success": bool(done), "env_steps": t, **extra,
                }) + "\n")

            # Save a replay video of the episode
            if cfg.save_video:
                rollout_dir = os.path.join(cfg.local_log_dir, run_id + "_videos")
                save_rollout_video(
                    replay_images, total_episodes,
                    success=done, task_description=task_description,
                    log_file=log_file,
                    rollout_dir=rollout_dir,
                )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

    log_file.close()

if __name__ == "__main__":
    eval_libero()
