# Final official evaluation on downloaded weights, for a free Kaggle (or Colab) GPU notebook.
# Run ONCE on the model already selected on validation. Paste into a notebook cell by cell.
#
# 1. Kaggle: New Notebook -> Settings -> Accelerator: GPU (T4 x2 or P100), Internet: On.
# 2. Upload the weights zip (from `overmind model download-checkpoint <id> --output weights.zip`)
#    as a Kaggle dataset, or download it in the notebook with the presigned link.

# --- cell 1: install
# !pip install -q "lm-eval[vllm]" safetensors

# --- cell 2: unpack weights
# !mkdir -p /kaggle/working/model && unzip -q /kaggle/input/<your-dataset>/weights.zip -d /kaggle/working/model
# !ls -R /kaggle/working/model | head -30     # find the folder that holds config.json
MODEL_DIR = "/kaggle/working/model"           # <- set to the folder containing config.json

# --- cell 3: official eval, raw mode (the task exactly as shipped)
# T4/P100 don't support bfloat16 -> float16. Results are greedy, so this does not change the method.
# !lm_eval --model vllm --model_args pretrained={MODEL_DIR},dtype=float16,gpu_memory_utilization=0.85,max_model_len=4096 \
#     --tasks gsm8k --num_fewshot 5 --batch_size auto --gen_kwargs max_gen_toks=1024 \
#     --log_samples --output_path /kaggle/working/results_raw
# If vLLM fails on the old GPU, swap to the HF backend:
#   --model hf --model_args pretrained={MODEL_DIR},dtype=float16 --batch_size 16

# --- cell 4: same, chat mode (only if organizers verify with the chat template)
# !lm_eval --model vllm --model_args pretrained={MODEL_DIR},dtype=float16,gpu_memory_utilization=0.85,max_model_len=4096 \
#     --tasks gsm8k --num_fewshot 5 --batch_size auto --gen_kwargs max_gen_toks=1024 \
#     --apply_chat_template --fewshot_as_multiturn \
#     --log_samples --output_path /kaggle/working/results_chat

# --- cell 5: params + score (upload scripts/score.py or paste it)
# !python score.py {MODEL_DIR} /kaggle/working/results_raw
# !python score.py {MODEL_DIR} /kaggle/working/results_chat
# Download /kaggle/working/results_* and keep them for verification.
