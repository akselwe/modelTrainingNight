#!/usr/bin/env python3
"""Build data/val_eval_5shot_chat.jsonl: the Overmind eval dataset for generate-mode runs.

Each row's input.messages is the exact lm-eval chat-mode context (--apply_chat_template
--fewshot_as_multiturn, 5 shots from train-minus-val, fewshot seed 1234) that lm-eval sent for
that val doc in the r1 run, so Overmind generate-eval sees the same prompt as our lm-eval setup.
expected_output is the gold final number from data/val_eval.jsonl.

Checks (exit 1 on failure): 1,000 docs, 11 messages each, last user turn == val_eval input,
every shot question comes from data/train_fewshot_raw.jsonl and none from data/val_raw.jsonl.

Usage (repo root): python3 -I scripts/build_val_eval_5shot.py
"""
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "verification/r1-qwen3-4b/samples_gsm8k_val_2026-10-05T23-04-56.781655.jsonl"
VAL_EVAL = ROOT / "data/val_eval.jsonl"
VAL_RAW = ROOT / "data/val_raw.jsonl"
TRAIN_FEWSHOT = ROOT / "data/train_fewshot_raw.jsonl"
OUT = ROOT / "data/val_eval_5shot_chat.jsonl"


def q(text):
    return f"Question: {text}\nAnswer:"


def main():
    contexts = {}
    for line in SAMPLES.open():
        row = json.loads(line)
        if row["filter"] != "strict-match":
            continue
        arg = row["arguments"]["gen_args_0"]["arg_0"]
        contexts[row["doc_id"]] = json.loads(arg[0] if isinstance(arg, list) else arg)

    val = [json.loads(line) for line in VAL_EVAL.open()]
    val_q = {q(json.loads(line)["question"]) for line in VAL_RAW.open()}
    train_q = {q(json.loads(line)["question"]) for line in TRAIN_FEWSHOT.open()}

    errors = []
    if sorted(contexts) != list(range(len(val))) or len(val) != 1000:
        errors.append(f"doc ids/rows mismatch: {len(contexts)} contexts, {len(val)} val rows")
    rows = []
    for doc_id, ref in enumerate(val):
        msgs = contexts.get(doc_id, [])
        roles = [m["role"] for m in msgs]
        if roles != ["user", "assistant"] * 5 + ["user"]:
            errors.append(f"doc {doc_id}: roles {roles}")
        if not msgs or msgs[-1]["content"] != ref["input"]:
            errors.append(f"doc {doc_id}: last user turn != val_eval input")
        for m in msgs[:-1]:
            if m["role"] == "user" and (m["content"] in val_q or m["content"] not in train_q):
                errors.append(f"doc {doc_id}: shot question not from train-minus-val")
        rows.append({
            "input": {"messages": [{"role": m["role"], "content": m["content"]} for m in msgs]},
            "expected_output": ref["expected_output"],
            "doc_id": doc_id,
        })
    if errors:
        print("\n".join(errors[:20]))
        sys.exit(1)
    with OUT.open("x") as f:  # never overwrite an existing file
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT.relative_to(ROOT)}")
    print(f"sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
