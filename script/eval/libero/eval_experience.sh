#!/usr/bin/env bash
# LIBERO evaluation of the experience-retrieval branch: one server, one client per condition (sessions keep their
# memory, stores and RNG apart), all on the same GPU so the runs pair up.
#   none        base policy (adapter attached, no references)
#   demo        references from the demo library only
#   online      library + this run's own episodes, recorded as they finish (successes become references)
#   unaligned   control: the demo references, but a random step of each instead of the aligned one
# With TRIALS_PER_INIT > 1 every initial state is repeated (init_major order), so `online` can reuse its own success
# on the same scene; analyze with script/eval/libero/analyze_experience.py.
#
#   CKPT_PATH=... ADAPTER=log/experience/libero_goal_no_noops/main/adapter_last.pt \
#   LIBRARY=cache/experience/libero_goal_no_noops/library.pt SUITE=libero_goal GPU=0 \
#     bash script/eval/libero/eval_experience.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export MKL_INTERFACE_LAYER=GNU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MEMVLA_LLAMA2_7B_PATH=${MEMVLA_LLAMA2_7B_PATH:-NousResearch/Llama-2-7b-hf} PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
py=${PYTHON_BIN:-python}
ckpt=${CKPT_PATH:?Set CKPT_PATH}; adapter=${ADAPTER:?Set ADAPTER}; library=${LIBRARY:?Set LIBRARY}
suite=${SUITE:-libero_goal}; gpu=${GPU:-0}; port=${PORT:-6851}
trials=${TRIALS:-10}; trials_per_init=${TRIALS_PER_INIT:-1}; seed_offset=${SEED_OFFSET:-0}
conditions=${CONDITIONS:-"none demo online unaligned"}
tag=${TAG:-exp}; out=${OUT_DIR:-./log/eval_experience/${suite}_${tag}}
extra_client_args=${CLIENT_ARGS:-}  # e.g. LIBERO-plus: --group_file ... --meta_instruction True
mkdir -p "$out"

CUDA_VISIBLE_DEVICES=$gpu "$py" deploy.py --saved_model_path "$ckpt" --unnorm_key "${suite}_no_noops" \
  --cfg_scale 1.5 --port "$port" --action_chunking --action_chunking_window 8 --use_bf16 \
  --exp_adapter "$adapter" --exp_library "$library" --exp_record_dir "$out/recorded" >"$out/server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
echo "Waiting for the server (pid $server) on port $port ..."
until curl -s -o /dev/null "http://localhost:$port/"; do
  kill -0 "$server" 2>/dev/null || { echo "Server died, see $out/server.log"; exit 1; }
  sleep 15
done

client() {  # name, condition, sources, record
  "$py" evaluation/libero/eval_libero.py --task_suite_name "$suite" --num_trials_per_task "$trials" \
    --trials_per_init "$trials_per_init" --seed_offset "$seed_offset" --port "$port" --local_log_dir "$out/$1" \
    --run_id_note "$1" --session_id "$1" --episode_api True --exp_condition "$2" --exp_sources "$3" \
    --exp_record "$4" --save_video False --agentview_only True $extra_client_args >"$out/$1.log" 2>&1
}
pids=()
for c in $conditions; do
  case $c in
    none)      client none none demo,rollout False & pids+=($!) ;;
    demo)      client demo real demo False & pids+=($!) ;;
    online)    client online real demo,rollout True & pids+=($!) ;;
    online_only) client online_only real rollout True & pids+=($!) ;;
    unaligned) client unaligned unaligned demo False & pids+=($!) ;;
    *) echo "Unknown condition $c"; exit 1 ;;
  esac
done
for pid in "${pids[@]}"; do wait "$pid"; done

runs=()
for c in $conditions; do runs+=("$c=$(ls "$out/$c"/*_results.jsonl | head -n1)"); done
"$py" script/eval/libero/analyze_experience.py --baseline none --runs "${runs[@]}" | tee "$out/summary.txt"
