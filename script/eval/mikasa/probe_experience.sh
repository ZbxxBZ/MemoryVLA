#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export MKL_INTERFACE_LAYER=GNU
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MEMVLA_LLAMA2_7B_PATH=${MEMVLA_LLAMA2_7B_PATH:-NousResearch/Llama-2-7b-hf}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Keep client->server requests off any HTTP(S) proxy in the environment (a proxied localhost call returns 502)
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

python_bin=${PYTHON_BIN:-python}
ckpt_path=${CKPT_PATH:?Set CKPT_PATH to memvla-mikasa.pt}
read -r -a task_ids <<< "${TASK_IDS:-1 2 3}"
num_inits=${NUM_INITS:-10}
collect_trials=${COLLECT_TRIALS:-5}
eval_trials=${EVAL_TRIALS:-2}
port=${PORT:-2345}
exp_k=${EXP_K:-1}
exp_store_dir=${EXP_STORE_DIR:-./exp_store/mikasa}
log_dir=${LOG_DIR:-./log/probe_experience/mikasa}
eval_modes=(none other_init_success same_init_success same_init_failure)

expected_ckpt_bytes=33507444130
if [[ ! -f "$ckpt_path" || $(stat -c %s "$ckpt_path") -ne $expected_ckpt_bytes ]]; then
  echo "Checkpoint incomplete: $ckpt_path (expected $expected_ckpt_bytes bytes)" >&2
  exit 1
fi
mkdir -p "$log_dir"

"$python_bin" deploy.py \
  --saved_model_path "$ckpt_path" \
  --unnorm_key mikasa_dataset \
  --cfg_scale 1.5 \
  --port "$port" \
  --use_bf16 \
  --preserve_unmasked_actions \
  --action_chunking \
  --action_chunking_window 4 \
  --exp_store_dir "$exp_store_dir" >"$log_dir/deploy.log" 2>&1 &
deploy_pid=$!
trap 'kill "$deploy_pid" 2>/dev/null || true' EXIT

until curl -s -o /dev/null "http://127.0.0.1:${port}/"; do
  kill -0 "$deploy_pid" 2>/dev/null || { tail -80 "$log_dir/deploy.log"; exit 1; }
  sleep 10
done

for task_id in "${task_ids[@]}"; do
  "$python_bin" evaluation/mikasa/eval_mikasa.py \
    --task_id "$task_id" --num_inits "$num_inits" --trials_per_init "$collect_trials" \
    --seed_offset 0 --exp_mode none --record_experience --run_id_note collect \
    --log_dir "$log_dir" --port "$port"
done

for task_id in "${task_ids[@]}"; do
  for mode in "${eval_modes[@]}"; do
    "$python_bin" evaluation/mikasa/eval_mikasa.py \
      --task_id "$task_id" --num_inits "$num_inits" --trials_per_init "$eval_trials" \
      --seed_offset 1000 --exp_mode "$mode" --exp_k "$exp_k" --run_id_note "eval-${mode}" \
      --log_dir "$log_dir" --port "$port"
  done
done

"$python_bin" script/eval/libero/analyze_probe.py --log_dir "$log_dir"
