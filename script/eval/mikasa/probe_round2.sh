#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export MKL_INTERFACE_LAYER=GNU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# Keep client->server requests off any HTTP(S) proxy in the environment (a proxied localhost call returns 502)
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export MEMVLA_LLAMA2_7B_PATH=${MEMVLA_LLAMA2_7B_PATH:-NousResearch/Llama-2-7b-hf} PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
py=${PYTHON_BIN:-python}
ckpt=${CKPT_PATH:?Set CKPT_PATH to memvla-mikasa.pt}
profile=${PROFILE:-full}; read -r -a tasks <<< "${TASK_IDS:-1 2 3}"
inits=${NUM_INITS:-30}; collect_trials=${COLLECT_TRIALS:-5}; eval_trials=${EVAL_TRIALS:-4}
diag_inits=${DIAG_INITS:-10}; port=${PORT:-2345}; k=${EXP_K:-1}
store=${EXP_STORE_DIR:-./exp_store/mikasa_r2}; logs=${LOG_DIR:-./log/probe_experience/mikasa_r2}
diag_dir=${DIAG_DIR:-${logs}_diag}; r1=${R1_LOG_DIR:-./log/probe_experience/mikasa_aligned}
done_dir="$logs/.done"; rc_tasks=(2 3)
case "$profile" in
  full)
    stages=("none 1000 all" "none 2000 all" "other_init_success 1000 all" "same_init_failure 1000 all" "other_task_success 1000 all" "noise 1000 all" "other_init_success_same_color 1000 rc" "other_init_success_diff_color 1000 rc" "same_init_success 1000 all")
    diag_modes=(other_init_success same_init_failure other_task_success noise) ;;
  min)
    stages=("none 1000 all" "none 2000 all" "noise 1000 all" "other_init_success_same_color 1000 rc" "other_init_success_diff_color 1000 rc")
    diag_modes=(noise) ;;
  *) echo "PROFILE must be full or min" >&2; exit 1 ;;
esac
diag_rc=(other_init_success_same_color other_init_success_diff_color)
[[ -f "$ckpt" && $(stat -c %s "$ckpt") -eq 33507444130 ]] || { echo "Checkpoint missing/incomplete: $ckpt" >&2; exit 1; }
mkdir -p "$logs" "$diag_dir" "$done_dir"; server_pid=""
start_server() {
  local logfile=$1; shift
  # Checkpoint loading peaks at tens of GB of host RAM; serialize loads across parallel runs to avoid OOM
  exec 9>"$logs/.load.lock"; flock 9
  "$py" deploy.py --saved_model_path "$ckpt" --unnorm_key mikasa_dataset --cfg_scale 1.5 --port "$port" \
    --use_bf16 --preserve_unmasked_actions --action_chunking --action_chunking_window 4 --exp_store_dir "$store" "$@" \
    >"$logfile" 2>&1 9>&- & server_pid=$!
  until curl -s -o /dev/null "http://127.0.0.1:$port/"; do
    kill -0 "$server_pid" 2>/dev/null || { tail -80 "$logfile"; exit 1; }; sleep 10
  done
  flock -u 9; exec 9>&-
}
stop_server() { if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; server_pid=""; fi; }
trap stop_server EXIT
tasks_for() { local t; for t in "${tasks[@]}"; do if [[ "$1" == all || " ${rc_tasks[*]} " == *" $t "* ]]; then echo "$t"; fi; done; }
run_eval() {
  local task=$1 mode=$2 off=$3 trials=$4 ninit=$5 out=$6 tag=$7; shift 7
  local marker="$done_dir/${tag}_task${task}_${mode}_${off}"
  [[ -f "$marker" ]] && { echo "Skip $marker"; return; }
  local t0=$SECONDS
  "$py" evaluation/mikasa/eval_mikasa.py --task_id "$task" --num_inits "$ninit" --trials_per_init "$trials" \
    --seed_offset "$off" --exp_mode "$mode" --exp_k "$k" --run_id_note "${tag}-${mode}-${off}" --log_dir "$out" --port "$port" "$@"
  printf "%s\t%s\t%s\t%s\t%s\n" "$tag" "$task" "$mode" "$off" "$((SECONDS-t0))" >>"$logs/timing.tsv"
  touch "$marker"
}
analyze() { "$py" script/eval/mikasa/analyze_round2.py --log_dir "$logs" >"$logs/analysis_latest_${port}.txt" 2>&1 || echo "WARN: analysis failed"; }
start_server "$logs/deploy_${port}.log"
for task in "${tasks[@]}"; do run_eval "$task" none 0 "$collect_trials" "$inits" "$logs" collect --record_experience; done
if [[ "${COLLECT_ONLY:-0}" == 1 ]]; then exit 0; fi
stop_server; start_server "$logs/deploy_${port}.log"
for stage in "${stages[@]}"; do
  read -r mode off scope <<<"$stage"
  for task in $(tasks_for "$scope"); do run_eval "$task" "$mode" "$off" "$eval_trials" "$inits" "$logs" eval; done
  if [[ "$mode" == none && "$off" == 1000 && "${SKIP_REPRO:-0}" != 1 && -d "$r1" ]]; then
    "$py" script/eval/mikasa/check_repro_round2.py --round1 "$r1" --round2 "$logs" || { echo "Reproducibility check failed" >&2; exit 1; }
  fi
  analyze
done
stop_server
if [[ "${RUN_DIAG:-1}" == 1 ]]; then
  start_server "$diag_dir/deploy_${port}.log" --exp_diag --exp_diag_path "$diag_dir/diag_steps_${port}.jsonl"
  for mode in "${diag_modes[@]}"; do for task in "${tasks[@]}"; do run_eval "$task" "$mode" 1000 1 "$diag_inits" "$diag_dir" diag; done; done
  for mode in "${diag_rc[@]}"; do for task in $(tasks_for rc); do run_eval "$task" "$mode" 1000 1 "$diag_inits" "$diag_dir" diag; done; done
  stop_server
  "$py" script/eval/mikasa/analyze_diag.py --diag_dir "$diag_dir" >"$diag_dir/diag_summary.txt" 2>&1 || echo "WARN: diagnostic analysis failed"
fi
analyze
echo "All done: $logs/analysis_latest_${port}.txt; timing=$logs/timing.tsv; diag=$diag_dir/diag_summary.txt"
