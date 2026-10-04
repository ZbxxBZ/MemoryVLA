#!/usr/bin/env bash
# Offline pipeline of the experience-retrieval branch:
#   cache raw tokens of the demos (one shard per GPU) -> demo experience library -> train the adapter (real references)
#   and the control adapter (unaligned references) in parallel.
# Finished stages leave markers in $WORK_DIR/.done and are skipped on re-runs; a cache shard also resumes by episode.
# An existing token cache of the task-memory branch (same layout, one frame per policy call) can be passed as
# TOKENS_DIR to skip the cache stage.
#
#   CKPT_PATH=/path/to/run/checkpoints/x.pt DATA_ROOT=/path/to/libero-rlds DATASET=libero_goal_no_noops GPUS="0 1" \
#     bash script/train/experience/run_pipeline.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export MKL_INTERFACE_LAYER=GNU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MEMVLA_LLAMA2_7B_PATH=${MEMVLA_LLAMA2_7B_PATH:-NousResearch/Llama-2-7b-hf} PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
py=${PYTHON_BIN:-python}
ckpt=${CKPT_PATH:?Set CKPT_PATH to the MemoryVLA checkpoint}
dataset=${DATASET:-libero_goal_no_noops}
work=${WORK_DIR:-./cache/experience/$dataset}; logs=${LOG_DIR:-./log/experience/$dataset}
tokens=${TOKENS_DIR:-$work/tokens}
read -r -a gpus <<< "${GPUS:-0}"
steps=${TRAIN_STEPS:-10000}
extra_train_args=${TRAIN_ARGS:-}
tool=script/train/experience
mkdir -p "$work/.done" "$logs"
todo() { if [[ -f "$work/.done/$1" ]]; then echo "Skip $1"; return 1; fi; }
mark() { touch "$work/.done/$1"; }

if [[ -z "${TOKENS_DIR:-}" ]] && todo cache; then
  data=${DATA_ROOT:?Set DATA_ROOT to the directory that contains $dataset/}
  pids=()
  for i in "${!gpus[@]}"; do
    # Model loads are serialized by a file lock inside cache_tokens.py (each load peaks at ~60 GB host RAM)
    CUDA_VISIBLE_DEVICES=${gpus[$i]} "$py" "$tool/cache_tokens.py" --ckpt "$ckpt" --data_root_dir "$data" \
      --dataset "$dataset" --out_dir "$tokens" --shard_id "$i" --num_shards "${#gpus[@]}" \
      >"$logs/cache_shard$i.log" 2>&1 & pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  mark cache
fi

if todo library; then
  "$py" "$tool/build_library.py" --cache_dir "$tokens" --out "$work/library.pt" 2>&1 | tee "$logs/build_library.txt"
  mark library
fi

train() {  # name, reference content, gpu
  CUDA_VISIBLE_DEVICES=$3 "$py" "$tool/train_experience.py" --ckpt "$ckpt" --cache_dir "$tokens" \
    --out_dir "$logs/$1" --ref_content "$2" --steps "$steps" $extra_train_args >"$logs/train_$1.log" 2>&1
}
if todo train; then
  second_gpu=${gpus[$(( ${#gpus[@]} > 1 ? 1 : 0 ))]}
  train main real "${gpus[0]}" & main_pid=$!
  train control unaligned "$second_gpu" & control_pid=$!
  wait "$main_pid"; wait "$control_pid"
  mark train
fi
echo "Done: adapter=$logs/main/adapter_last.pt control=$logs/control/adapter_last.pt library=$work/library.pt"
