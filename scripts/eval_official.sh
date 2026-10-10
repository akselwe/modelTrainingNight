#!/usr/bin/env bash
# Official-style evaluation on DOWNLOADED weights (this is what gets verified).
# Needs one GPU (e.g. Modal / RunPod / Colab). Usage:
#   bash eval_official.sh ./weights_dir raw      # task exactly as shipped (no chat template)
#   bash eval_official.sh ./weights_dir chat     # same task, chat template + multi-turn few-shot
set -euo pipefail
MODEL_DIR=${1:?path to model weights}
MODE=${2:-raw}
pip install -q "lm-eval[vllm]"

EXTRA=""
if [ "$MODE" = "chat" ]; then
  EXTRA="--apply_chat_template --fewshot_as_multiturn"
fi

# Rules: EleutherAI gsm8k task, 5-shot, greedy, 1 response, max 1024 generated tokens, strict-match.
# NB: lm-eval defaults to 256 generated tokens - we raise it to the allowed 1024.
lm_eval --model vllm \
  --model_args pretrained="$MODEL_DIR",dtype=bfloat16,gpu_memory_utilization=0.85 \
  --tasks gsm8k --num_fewshot 5 --batch_size auto \
  --gen_kwargs max_gen_toks=1024 \
  $EXTRA \
  --log_samples --output_path "results/$(basename "$MODEL_DIR")_$MODE"

# Read "exact_match,strict-match" from the printed table. That is the accuracy to submit.
python "$(dirname "$0")/score.py" "$MODEL_DIR" results/"$(basename "$MODEL_DIR")_$MODE"
