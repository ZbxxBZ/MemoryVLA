#!/bin/bash
# Pilot: can prefilling cross-episode experience into MemoryVLA's memory improve LIBERO success without training?
#   Phase 1 (collect): exp_mode=none, several rollouts per initial state, recorded into the experience store.
#   Phase 2 (eval):    A/B/C/D experience modes on the same initial states with fresh, shared seeds.
# Summarize with: python script/eval/libero/analyze_probe.py --log_dir ${log_dir}
set -euo pipefail
export MKL_INTERFACE_LAYER=GNU

ckpt_path=PATH_TO_CKPT

task_suite_name=libero_10
task_ids=(0 1 2)            # pick tasks with ~40%-70% success
num_inits=10                # use the first N initial states of each task
gpu_id=0
action_chunking_window=8
timestep_stride=1           # set to ${action_chunking_window} to align memory timesteps with training frames

collect_trials=5            # phase 1 rollouts per initial state
eval_trials=2               # phase 2 rollouts per initial state and mode
eval_seed_offset=1000       # phase 2 must not reuse phase 1 seeds
eval_modes=(none other_init_success same_init_success same_init_failure)
exp_k=1

exp_store_dir=./exp_store/${task_suite_name}
log_dir=./log/probe_experience/${task_suite_name}
unnorm_key="${task_suite_name}_no_noops"

find_free_port() {
  local min=${1:-2000}
  local max=${2:-30000}
  local port
  for ((i=0; i<1000; i++)); do
    port=$(shuf -i"${min}"-"${max}" -n1)
    if ! lsof -iTCP:"${port}" -sTCP:LISTEN &>/dev/null; then
      echo "${port}"
      return 0
    fi
  done
  echo "ERROR: not found free port in range ${min}-${max}" >&2
  return 1
}

export CUDA_VISIBLE_DEVICES=${gpu_id}
port=$(find_free_port)
mkdir -p "${log_dir}"

echo ">>> Starting server on port ${port}"
python deploy.py \
  --saved_model_path ${ckpt_path} \
  --unnorm_key ${unnorm_key} \
  --cfg_scale 1.5 \
  --port ${port} \
  --action_chunking \
  --action_chunking_window ${action_chunking_window} \
  --timestep_stride ${timestep_stride} \
  --exp_store_dir ${exp_store_dir} &
DEPLOY_PID=$!
trap 'kill ${DEPLOY_PID} 2>/dev/null || true' EXIT

# Wait until the server answers (any HTTP status) instead of a fixed sleep
until curl -s -o /dev/null "http://localhost:${port}/"; do
  kill -0 ${DEPLOY_PID} 2>/dev/null || { echo "Server exited during loading"; exit 1; }
  sleep 30
done

for task_id in "${task_ids[@]}"; do
  echo ">>> [phase 1] collect task ${task_id}"
  python evaluation/libero/eval_libero.py \
    --task_suite_name ${task_suite_name} \
    --spcial_task_id ${task_id} \
    --num_trials_per_task ${num_inits} \
    --trials_per_init ${collect_trials} \
    --episode_api True \
    --record_experience True \
    --exp_mode none \
    --seed_offset 0 \
    --save_video False \
    --run_id_note collect \
    --local_log_dir ${log_dir} \
    --port ${port}
done

for task_id in "${task_ids[@]}"; do
  for mode in "${eval_modes[@]}"; do
    echo ">>> [phase 2] task ${task_id}, mode ${mode}"
    python evaluation/libero/eval_libero.py \
      --task_suite_name ${task_suite_name} \
      --spcial_task_id ${task_id} \
      --num_trials_per_task ${num_inits} \
      --trials_per_init ${eval_trials} \
      --episode_api True \
      --exp_mode ${mode} \
      --exp_k ${exp_k} \
      --seed_offset ${eval_seed_offset} \
      --save_video False \
      --run_id_note eval-${mode} \
      --local_log_dir ${log_dir} \
      --port ${port}
  done
done

python script/eval/libero/analyze_probe.py --log_dir ${log_dir}
echo "All done!"
