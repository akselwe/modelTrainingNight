#!/usr/bin/env bash
# Evaluates a model served by Overmind (no local GPU needed). Chat-template mode only,
# because Overmind serves chat completions only.
# Usage (from repo root):
#   OVERMIND_API_KEY=... bash scripts/eval_overmind_endpoint.sh <model-ref> val [limit]   # 1,000 validation rows (use this in the loop)
#   OVERMIND_API_KEY=... bash scripts/eval_overmind_endpoint.sh <model-ref> test          # official test - ONLY for the final selected model
# <model-ref> e.g. ft-9635f347-qwen3-4b. First call may cold-boot the model (2-7 min).
set -euo pipefail
MODEL_ID=${1:?overmind model reference, e.g. ft-9635f347-qwen3-4b}
SPLIT=${2:-val}
LIMIT=${3:-}
# Isolated uv tool env (lm-eval 0.4.x on Python 3.12); doesn't touch the project manifest.
LM_EVAL=(uvx --python 3.12 --from "lm-eval[api]==0.4.13" lm_eval)
# uv-managed Python lacks a CA bundle on macOS -> use the system one (verification stays on).
[ -z "${SSL_CERT_FILE:-}" ] && [ -f /etc/ssl/cert.pem ] && export SSL_CERT_FILE=/etc/ssl/cert.pem
# Key: env var, else the project-scoped key written by `overmind sync`.
OVERMIND_API_KEY=${OVERMIND_API_KEY:-$(python3 -I -c "import tomllib;print(tomllib.load(open('.overmind/credentials.toml','rb'))['api-key'])")}
export OPENAI_API_KEY="$OVERMIND_API_KEY"

if [ "$SPLIT" = "val" ]; then TASK="--include_path scripts/gsm8k_val --tasks gsm8k_val"
elif [ "$SPLIT" = "test" ]; then TASK="--tasks gsm8k"
else echo "split must be val or test"; exit 1; fi
LIM=""; [ -n "$LIMIT" ] && LIM="--limit $LIMIT"

# Qwen 3 / 3.5: thinking tokens count against the 1,024-token budget. Check the logged samples
# for long reasoning before the answer; if present, report it (see SPEC.md section 5).
"${LM_EVAL[@]}" --model local-chat-completions \
  --model_args model="$MODEL_ID",base_url=https://api.overmindlab.ai/api/v1/chat/completions,num_concurrent=16,max_retries=5,tokenized_requests=False \
  $TASK --num_fewshot 5 --apply_chat_template --fewshot_as_multiturn \
  --gen_kwargs '{"max_gen_toks": 1024, "temperature": 0, "do_sample": false, "chat_template_kwargs": {"enable_thinking": false}}' \
  $LIM --log_samples --output_path "results/endpoint_${SPLIT}_$MODEL_ID"
