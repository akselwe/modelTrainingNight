#!/usr/bin/env python3
"""Runde 4: build SFT targets from verified Qwen3 solutions to the GSM8K train questions.

Why: untouched Qwen3-8B gets the right number on 94.3% of val, but fine-tuning on the terse
official solutions pulls it down to ~89.8%. Training on verified solutions written by a model
of the same family should teach the `#### N` format while keeping a Qwen-style reasoning.
The plan was the student's own solutions; the served qwen/qwen3-8b stopped answering
(HTTP 502/404, 2026-10-10), so round 4 used qwen/qwen3-32b as the teacher (the default below).

Steps (repo root):
  python3 -I scripts/self_distill.py gen --attempt 0 --max-cost 3.0          # greedy, all 6,473
  python3 -I scripts/self_distill.py gen --attempt 1 --temperature 0.7 --max-cost 1.0  # only still-unsolved
  python3 -I scripts/self_distill.py build                                     # -> data/train_selfdistill_sft.jsonl
  (add --style native to gen/build for the zero-shot variant -> data/train_selfdistill_native_sft.jsonl;
   data/train_selfdistill_sft.jsonl was built before the `####` heading fix below, so a rebuild changes 6 rows)
  python3 -I scripts/self_distill.py combine   # both styles, 2 solutions per question -> data/train_selfdistill_combined_sft.jsonl

Prompt (--style fewshot, default) = lm-eval-style 5-shot multiturn chat (5 other train examples
with official answers, seeded per question). --style native = zero-shot, the question plus a
one-line format hint, so the teacher writes in its own step-by-step style. Both put
" /no_think" on the last user turn (served provider ignores enable_thinking). The SFT rows
always use the plain "Question: ...\nAnswer:" prompt, never the hint. An answer is kept only if
its final number equals the official answer; its final line is then normalised to `#### <gold>`.
Unsolved questions fall back to the official solution. Only GSM8K train is used; val and test are never touched.
The API key is read from .overmind/credentials.toml and never printed.
"""
import argparse
import concurrent.futures as cf
import json
import pathlib
import random
import re
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from served_eval import ROOT, UNTIL, api_key, call, format_free_ok  # noqa: E402

TRAIN = ROOT / "data/train_fewshot_raw.jsonl"
STYLES = {
    "fewshot": (ROOT / "results/round4/distill", ROOT / "data/train_selfdistill_sft.jsonl"),
    "native": (ROOT / "results/round4/distill_native", ROOT / "data/train_selfdistill_native_sft.jsonl"),
}
OUT, SFT = STYLES["fewshot"]
NATIVE_HINT = "\n\nSolve it step by step. Put the final numeric answer on its own last line as: #### <number>"
DEFAULT_MODEL = "qwen/qwen3-32b"  # the teacher used in round 4; served qwen/qwen3-8b returned HTTP 502/404 on 2026-10-10


def load_train():
    rows = [json.loads(line) for line in TRAIN.open()]
    for r in rows:
        r["gold"] = r["answer"].split("####")[-1].strip().replace(",", "")
    return rows


def user(q):
    return {"role": "user", "content": f"Question: {q}\nAnswer:"}


def messages(rows, i, style="fewshot"):
    if style == "native":
        msg = user(rows[i]["question"])
        msg["content"] += NATIVE_HINT + " /no_think"
        return [msg]
    rng = random.Random(1000 + i)
    shots = rng.sample([j for j in range(len(rows)) if j != i], 5)
    msgs = []
    for j in shots:
        msgs += [user(rows[j]["question"]), {"role": "assistant", "content": rows[j]["answer"]}]
    msgs.append(user(rows[i]["question"]))
    msgs[-1]["content"] += " /no_think"
    return msgs


def attempt_path(k):
    return OUT / f"gen_attempt{k}.jsonl"


def records(k):
    p = attempt_path(k)
    recs = {}
    if p.exists():
        for line in p.open():
            r = json.loads(line)
            recs[r["idx"]] = r
    return recs


def solved_upto(k):
    """Indices with a verified-correct answer in attempts < k."""
    out = set()
    for a in range(k):
        out |= {i for i, r in records(a).items() if r.get("ok")}
    return out


def gen(args):
    rows = load_train()
    OUT.mkdir(parents=True, exist_ok=True)
    done = {i for i, r in records(args.attempt).items() if r.get("content") is not None}
    skip = solved_upto(args.attempt)
    todo = [i for i in range(len(rows)) if i not in done and i not in skip]
    if args.limit:
        todo = todo[:args.limit]
    spent = sum(r.get("cost") or 0 for r in records(args.attempt).values())
    key, lock = api_key(), threading.Lock()
    print(f"attempt {args.attempt}: {len(todo)} to run, spent so far ${spent:.4f}", flush=True)

    def one(i):
        nonlocal spent
        with lock:
            if spent >= args.max_cost:
                return None
        body = {"model": args.model, "messages": messages(rows, i, args.style), "max_tokens": 1024,
                "temperature": args.temperature, "stop": UNTIL,
                "chat_template_kwargs": {"enable_thinking": False}}
        if args.temperature > 0:
            body["seed"] = 7919 * args.attempt + i
        d = call(key, body)
        rec = {"idx": i, "gold": rows[i]["gold"], "model": args.model, "attempt": args.attempt, "temperature": args.temperature}
        if "error" in d:
            rec.update(error=d["error"], content=None, ok=False)
        else:
            c, u = d["choices"][0], d.get("usage") or {}
            content = c["message"].get("content") or ""
            rec.update(content=content, finish_reason=c.get("finish_reason"), provider=d.get("provider"),
                       completion_tokens=u.get("completion_tokens"), cost=u.get("cost"),
                       ok=c.get("finish_reason") == "stop" and "<think>" not in content
                       and format_free_ok(content, rows[i]["gold"]))
        with lock:
            spent += rec.get("cost") or 0
            with attempt_path(args.attempt).open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    n_ok = 0
    with cf.ThreadPoolExecutor(args.concurrency) as ex:
        for n, rec in enumerate(ex.map(one, todo), 1):
            n_ok += bool(rec and rec["ok"])
            if n % 250 == 0:
                print(f"  {n}/{len(todo)} done, ok {n_ok}, spent ${spent:.4f}", flush=True)
    recs = records(args.attempt)
    print(f"attempt {args.attempt}: {sum(r.get('ok', False) for r in recs.values())} ok of {len(recs)}, "
          f"errors {sum(r.get('content') is None for r in recs.values())}, spent ${spent:.4f}; "
          f"solved total {len(solved_upto(args.attempt + 1))}/{len(rows)}")
    return 0


# The final answer marker. Teachers also write `####` as a markdown heading ("#### Year 1:"),
# so cut at the LAST marker followed by a number, not at the first `####`.
# (Before 2026-10-10 21:40 the cut was at the first `####`; that truncated 6 rows of
# data/train_selfdistill_sft.jsonl and 93 native rows inside data/train_selfdistill_combined_sft.jsonl.)
FINAL = re.compile(r"#### \**\s*(?:Answer:\s*)?\$?(-?[0-9][0-9.,]*)")


def final_number_ok(content, gold):
    hits = FINAL.findall(content)
    if not hits:
        return True  # no marker: format_free_ok already checked \boxed{} / last number
    try:
        return abs(float(hits[-1].replace(",", "").rstrip(".")) - float(gold)) < 1e-6
    except ValueError:
        return False


def clean(content, gold):
    """Cut at the final `#### <number>`, demote any other `####` (headings) to `###`, then end
    with a clean `#### <gold>`; None if no reasoning is left."""
    hits = list(FINAL.finditer(content))
    body = content[:hits[-1].start()] if hits else content
    body = body.replace("####", "###").strip()
    return f"{body}\n#### {gold}" if len(body) >= 20 else None


def build(args):
    rows = load_train()
    attempts = sorted(int(p.stem.removeprefix("gen_attempt")) for p in OUT.glob("gen_attempt*.jsonl"))
    best = {}
    for k in attempts:
        for i, r in records(k).items():
            if r.get("ok") and i not in best and final_number_ok(r["content"], r["gold"]) \
                    and clean(r["content"], r["gold"]):
                best[i] = r
    stats = {"rows": len(rows), "own_answer": 0, "official_fallback": 0, "by_attempt": {}}
    with SFT.open("w") as f:
        for i, row in enumerate(rows):
            if i in best:
                target = clean(best[i]["content"], row["gold"])
                stats["own_answer"] += 1
                a = str(best[i]["attempt"])
                stats["by_attempt"][a] = stats["by_attempt"].get(a, 0) + 1
            else:
                target = row["answer"]
                stats["official_fallback"] += 1
            msgs = [user(row["question"]), {"role": "assistant", "content": target}]
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


def combine(args):
    """Both styles in one file (2 rows per question), shuffled with a fixed seed.
    A question that fell back to the official solution in both styles appears once."""
    rows = [json.loads(line) for line in STYLES["fewshot"][1].open()]
    native = [json.loads(line) for line in STYLES["native"][1].open()]
    assert len(rows) == len(native) and all(a["messages"][0] == b["messages"][0] for a, b in zip(rows, native))
    rows += [b for a, b in zip(rows[:], native) if b != a]
    random.Random(4).shuffle(rows)
    out = ROOT / "data/train_selfdistill_combined_sft.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows -> {out.relative_to(ROOT)}")
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--attempt", type=int, required=True)
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--concurrency", type=int, default=16)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--max-cost", type=float, required=True, help="stop starting new requests above this USD")
    b = sub.add_parser("build")
    sub.add_parser("combine")
    for sp in (g, b):
        sp.add_argument("--style", choices=sorted(STYLES), default="fewshot")
    a = p.parse_args()
    if a.cmd == "combine":
        return combine(a)
    global OUT, SFT
    OUT, SFT = STYLES[a.style]
    return gen(a) if a.cmd == "gen" else build(a)


if __name__ == "__main__":
    sys.exit(main())
