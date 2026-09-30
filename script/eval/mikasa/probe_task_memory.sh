#!/usr/bin/env bash
# MIKASA-Robo evaluation of the trained task memory (same protocol and seeds as round 2).
#   task model + demo library:     none (branch idle = base model), other_init_success (real), noise, other_task_success
#   control model + demo library:  other_init_success
#   task model + rollout library:  colour-matched / mismatched experiences on RememberColor (only with ROLLOUT_LIB)
# Finished (model, task, mode, seed) combinations leave markers in $LOG_DIR/.done and are skipped on re-runs.
#
#   CKPT_PATH=... TASK_MEM=log/task_memory/main/task_memory_last.pt CTRL_MEM=log/task_memory/control/task_memory_last.pt \
#     DEMO_LIB=cache/task_memory/demo_library.pt bash script/eval/mikasa/probe_task_memory.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export MKL_INTERFACE_LAYER=GNU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# Keep client->server requests off any HTTP(S) proxy in the environment (a proxied localhost call returns 502)
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export MEMVLA_LLAMA2_7B_PATH=${MEMVLA_LLAMA2_7B_PATH:-NousResearch/Llama-2-7b-hf} PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
py=${PYTHON_BIN:-python}
ckpt=${CKPT_PATH:?Set CKPT_PATH to memvla-mikasa.pt}
task_mem=${TASK_MEM:?Set TASK_MEM to the trained task-memory checkpoint}
ctrl_mem=${CTRL_MEM:?Set CTRL_MEM to the control (noise-trained) checkpoint}
demo_lib=${DEMO_LIB:?Set DEMO_LIB to the demo library from build_experiences.py}
rollout_lib=${ROLLOUT_LIB:-}
read -r -a tasks <<< "${TASK_IDS:-1 2 3}"
inits=${NUM_INITS:-30}; trials=${EVAL_TRIALS:-4}; port=${PORT:-2345}; k=${EXP_K:-1}
logs=${LOG_DIR:-./log/task_memory/eval}; done_dir="$logs/.done"; rc_tasks=(2 3 4)
baseline_2000=${BASELINE_2000:-1}  # also run the seed placebo (none@2000)
mkdir -p "$logs" "$done_dir"; server_pid=""

start_server() {
  local logfile=$1; shift
  # Checkpoint loading peaks at tens of GB of host RAM; serialize loads across parallel runs to avoid OOM
  exec 9>"$logs/.load.lock"; flock 9
  "$py" deploy.py --saved_model_path "$ckpt" --unnorm_key mikasa_dataset --cfg_scale 1.5 --port "$port" \
    --use_bf16 --preserve_unmasked_actions --action_chunking --action_chunking_window 4 "$@" \
    >"$logfile" 2>&1 9>&- & server_pid=$!
  until curl -s -o /dev/null "http://127.0.0.1:$port/"; do
    kill -0 "$server_pid" 2>/dev/null || { tail -80 "$logfile"; exit 1; }; sleep 10
  done
  flock -u 9; exec 9>&-
}
stop_server() { if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; server_pid=""; fi; }
trap stop_server EXIT
is_rc() { [[ " ${rc_tasks[*]} " == *" $1 "* ]]; }
run_eval() {  # model tag, task, mode, seed offset
  local marker="$done_dir/$1_task$2_$3_$4"
  [[ -f "$marker" ]] && { echo "Skip $marker"; return; }
  "$py" evaluation/mikasa/eval_mikasa.py --task_id "$2" --num_inits "$inits" --trials_per_init "$trials" \
    --seed_offset "$4" --exp_mode "$3" --exp_k "$k" --model_tag "$1" --run_id_note "$1-$3-$4" --log_dir "$logs" --port "$port"
  touch "$marker"
}

start_server "$logs/deploy_task_demo_${port}.log" --task_memory_ckpt "$task_mem" --task_exp_path "$demo_lib"
for task in "${tasks[@]}"; do
  run_eval task "$task" none 1000
  if [[ "$baseline_2000" == 1 ]]; then run_eval task "$task" none 2000; fi
  for mode in other_init_success noise other_task_success; do run_eval task "$task" "$mode" 1000; done
done
stop_server

start_server "$logs/deploy_control_demo_${port}.log" --task_memory_ckpt "$ctrl_mem" --task_exp_path "$demo_lib"
for task in "${tasks[@]}"; do run_eval control "$task" other_init_success 1000; done
stop_server

if [[ -n "$rollout_lib" ]]; then
  start_server "$logs/deploy_task_rollout_${port}.log" --task_memory_ckpt "$task_mem" --task_exp_path "$rollout_lib"
  for task in "${tasks[@]}"; do
    if is_rc "$task"; then
      for mode in other_init_success_same_color other_init_success_diff_color; do run_eval task_rollout "$task" "$mode" 1000; done
    fi
  done
  stop_server
fi

"$py" script/eval/mikasa/analyze_task_memory.py --log_dir "$logs" ${R2_LOG_DIR:+--round2_dir "$R2_LOG_DIR"} \
  >"$logs/analysis_${port}.txt" 2>&1 || echo "WARN: analysis failed"
echo "All done: $logs/analysis_${port}.txt"
