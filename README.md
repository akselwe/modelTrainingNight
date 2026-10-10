# Model Training Night: fine-tuning a small model on GSM8K

Fine-tuning small open models on grade-school maths word problems (GSM8K), built for Overmind's "Small Model, Big Maths" challenge at Model Training Night in San Francisco (SF Tech Week, 5–6 October 2026).

The challenge score was test accuracy minus model size in billions of parameters, so the task was to find the best accuracy per parameter, not just the best accuracy.

**Best result so far:** fine-tuning Qwen3-8B raised accuracy on 1,000 held-out questions from **81.9% to 89.7%** (strict match, same evaluation before and after). See [Results](#results-before-and-after-fine-tuning-qwen3-8b).

## What I did

- **Model choice from published results.** I picked Qwen3-4B for the single training run the free plan allowed, based on the base-model GSM8K numbers in the Qwen3 technical report ([arXiv:2505.09388](https://arxiv.org/abs/2505.09388), Table 8), weighing accuracy against parameter count.
- **Data and rules.** Training used only the official GSM8K train split with its original worked answers as targets. I carved a 1,000-question validation set out of the train split (seed 42) and checked it against the official file by SHA-256 and for overlap (`verification/data_sha256.txt`). The test split was never used for training or model selection.
- **Full fine-tune on Overmind.** Qwen3-4B, full fine-tune, 3 epochs, LR 1e-5, packed 4,096-token sequences (`verification/r1-qwen3-4b/run.json`).
- **Evaluation setup.** The official EleutherAI `lm-eval` GSM8K task (5-shot, greedy, max 1,024 generated tokens, strict match on `#### N`), pointed at the 1,000 held-out validation questions and run against the deployed model (`scripts/eval_overmind_endpoint.sh`, `scripts/gsm8k_val/`). Thinking mode was disabled because thinking tokens count against the budget.
- **Error analysis.** An automatic scan of every wrong answer (truncation, missing `####`, thinking leakage, wrong `<<calc>>` steps) plus a manual read of 50 errors (`verification/r1-qwen3-4b/error_breakdown_auto.txt`). Finding: the answer format was fully learned (0 truncated, 3 of 181 missing `####`); almost all remaining errors were reasoning errors (misread or skipped quantities), not arithmetic or formatting.

The event run scored 81.9% strict match on the 1,000 held-out questions. Qwen3-4B itself was not evaluated on the same setup before training (the platform could not serve the untouched model), so that number says nothing about improvement. The follow-up below does measure before and after.

### Follow-up after the event

After the event I kept going with the same rules to understand what actually moves accuracy:

- `scripts/served_eval.py` re-implements the lm-eval prompts and strict/flexible scoring against a served endpoint, verified row by row against lm-eval output (`selftest`).
- I compared untouched served models, then LoRA runs on Qwen3-8B and Qwen3-4B with different learning rates and epoch counts, screening on 200 validation questions before running the full 1,000 and using paired sign tests between variants.
- Main lessons: with the official terse answers as targets, *less* training did not preserve the model's own reasoning style, it only left the format half-learned; more training (higher LR or a second epoch) helped. Light LoRA and a 3-epoch full fine-tune gave the same result for the 4B model.
- I also documented platform behaviour that affected the experiments (ignored trainer keys, fractional epochs training zero steps, cancel requests not honoured during preparation).

### Results: before and after fine-tuning (Qwen3-8B)

Same 1,000 held-out validation questions, same evaluation (lm-eval 5-shot chat prompts, greedy, strict match on `#### N`), measured before and after training:

| Model | Training | Strict match | Evidence |
|---|---|---|---|
| Qwen3-8B, untouched | none | 81.9% | `results/round0/qwen3-8b_served_nothink_full/` |
| Qwen3-8B | LoRA (r=32), LR 1e-4, 1 epoch | 87.0% | `results/round1/A_qwen3-8b_lora_ep1/` |
| Qwen3-8B | LoRA (r=32), LR 1e-4, 2 epochs | **89.7%** (95% CI 87.8–91.6) | `results/round3/R3b_qwen3-8b_lora_lr1e-4_ep2/` |

- **+7.8 points from fine-tuning.** The 2-epoch run beats the 1-epoch run in a paired comparison on the same questions (48 vs 21 discordant answers, sign test p = 0.002).
- **No sign of overfitting to the selection set.** Variants were screened on questions 0–199; on questions 200–999, which were not used for selection, the 2-epoch run scores 90.0% vs 87.1%.
- **Most of the gain is format, not maths.** The untouched model already had the right number in 94.3% of answers but often dropped the required `#### N` line. Fine-tuning mainly taught the expected answer format while keeping most of the model's correct answers (877 of 943).

Per-run outputs, sample files and diagnostics are in `results/`.

## Stack

Python 3, EleutherAI `lm-eval`, Overmind (fine-tuning, deployment and inference API), Hugging Face safetensors metadata for exact parameter counts, `uv`. Built with Claude Code as a coding agent.

## Repository layout

| Path | Contents |
|---|---|
| `data/` | Train/validation split of the official GSM8K train set (SFT and lm-eval formats) |
| `scripts/` | Evaluation (lm-eval wrapper, served-endpoint eval and scorer), parameter counting, final-eval notebook |
| `verification/` | Data hashes and the full evidence for the event run (config, eval log, samples, error breakdown) |
| `results/` | Follow-up experiments: samples, summaries and diagnostics per run |

## Running the evaluation

Requires an Overmind API key in `.overmind/credentials.toml` (not committed) and a deployed model.

```bash
uv sync
# lm-eval on the 1,000 validation questions against a deployed model
bash scripts/eval_overmind_endpoint.sh <model-ref> val
# or the standalone evaluator (check that it matches lm-eval first)
python3 -I scripts/served_eval.py selftest
python3 -I scripts/served_eval.py run --model <model-ref> --label my_run --docs 0-999
```

## Data licence

GSM8K is © OpenAI, released under the MIT licence ([openai/grade-school-math](https://github.com/openai/grade-school-math)).
