#!/usr/bin/env bash
# Offline pipeline of the trained task memory:
#   inspect the dataset -> cache tokens (one shard per GPU) -> build the demo (and optional rollout) library
#   -> train the main branch (real experiences) and the control branch (noise) in parallel.
# Finished stages leave markers in $WORK_DIR/.done and are skipped on re-runs; a cache shard also resumes by episode.
#
#   CKPT_PATH=/path/to/memvla-mikasa.pt DATA_ROOT=/path/to/mikasa-rlds GPUS="0 1" \
#     bash script/train/task_memory/run_pipeline.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export MKL_INTERFACE_LAYER=GNU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MEMVLA_LLAMA2_7B_PATH=${MEMVLA_LLAMA2_7B_PATH:-NousResearch/Llama-2-7b-hf} PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
py=${PYTHON_BIN:-python}
ckpt=${CKPT_PATH:?Set CKPT_PATH to memvla-mikasa.pt}
data=${DATA_ROOT:?Set DATA_ROOT to the directory that contains mikasa_dataset/}
work=${WORK_DIR:-./cache/task_memory}; logs=${LOG_DIR:-./log/task_memory}
read -r -a gpus <<< "${GPUS:-0}"
rollout_store=${ROLLOUT_STORE:-}  # optional deploy.py ExperienceStore (e.g. exp_store/mikasa_r2) for colour conditions
steps=${TRAIN_STEPS:-10000}; task_map=${TASK_MAP:-}
extra_train_args=${TRAIN_ARGS:-}
tool=script/train/task_memory
mkdir -p "$work/.done" "$logs"
todo() { if [[ -f "$work/.done/$1" ]]; then echo "Skip $1"; return 1; fi; }
mark() { touch "$work/.done/$1"; }

if todo inspect; then
  "$py" "$tool/inspect_dataset.py" --data_root_dir "$data" --ckpt "$ckpt" ${task_map:+--task_map "$task_map"} \
    2>&1 | tee "$logs/inspect_dataset.txt"
  mark inspect
fi

if todo cache; then
  pids=()
  for i in "${!gpus[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpus[$i]} "$py" "$tool/cache_tokens.py" --ckpt "$ckpt" --data_root_dir "$data" \
      --out_dir "$work/tokens" --shard_id "$i" --num_shards "${#gpus[@]}" >"$logs/cache_shard$i.log" 2>&1 & pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  mark cache
fi

if todo library; then
  CUDA_VISIBLE_DEVICES=${gpus[0]} "$py" "$tool/build_experiences.py" --ckpt "$ckpt" --cache_dir "$work/tokens" \
    --out "$work/demo_library.pt" ${task_map:+--task_map "$task_map"} 2>&1 | tee "$logs/build_demo_library.txt"
  mark library
fi

if [[ -n "$rollout_store" ]] && todo rollout_library; then
  CUDA_VISIBLE_DEVICES=${gpus[0]} "$py" "$tool/build_experiences.py" --ckpt "$ckpt" --exp_store_dir "$rollout_store" \
    --out "$work/rollout_library.pt" 2>&1 | tee "$logs/build_rollout_library.txt"
  mark rollout_library
fi

train() {  # name, experience content, gpu
  CUDA_VISIBLE_DEVICES=$3 "$py" "$tool/train_task_memory.py" --ckpt "$ckpt" --cache_dir "$work/tokens" \
    --library "$work/demo_library.pt" --out_dir "$logs/$1" --exp_content "$2" --steps "$steps" $extra_train_args \
    >"$logs/train_$1.log" 2>&1
}
if todo train; then
  second_gpu=${gpus[$(( ${#gpus[@]} > 1 ? 1 : 0 ))]}
  train main real "${gpus[0]}" & main_pid=$!
  train control noise "$second_gpu" & control_pid=$!
  wait "$main_pid"; wait "$control_pid"
  mark train
fi
echo "Done: main=$logs/main/task_memory_last.pt control=$logs/control/task_memory_last.pt library=$work/demo_library.pt"
