# Instructions for coding agents

## Keep the README results current

`README.md` is the public face of this repo and shows the best measured result. Whenever a new training run beats the current best on the 1,000-question validation set:

1. Update the **Best result so far** line at the top of `README.md` (before → after, model name).
2. Add the run as a new row in the **Results: before and after fine-tuning** table, with the path to its folder in `results/`.
3. Only report numbers measured with the same evaluation as the existing rows (lm-eval 5-shot chat prompts, greedy, strict match on `#### N`, the same 1,000 validation questions). Never use the GSM8K test split.
4. State before and after for the same base model. Do not claim an improvement for a model whose untouched baseline was not measured.
5. Keep internal details out of the README: credits, costs, job or deployment IDs and API keys.

If a new run is not better, the README does not need to change.
