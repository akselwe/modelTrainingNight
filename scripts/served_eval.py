#!/usr/bin/env python3
"""GSM8K val eval of models *served* by the Overmind chat endpoint, scored exactly like lm-eval.

Why: Overmind generate-eval and /api/v1/chat/completions only accept served models
(see GET /api/v1/models) or project deployments, not undeployed catalog ids such as
Qwen/Qwen3-4B. For served models this script replays the exact lm-eval chat-mode contexts
(data/val_eval_5shot_chat.jsonl = r1's --apply_chat_template --fewshot_as_multiturn 5-shot
prompts), greedy, max 1,024 new tokens, stop = lm-eval `until`, and scores with a
re-implementation of lm-eval's strict-match / flexible-extract (validated by `selftest`).

Usage (repo root):
  python3 -I scripts/served_eval.py selftest
  python3 -I scripts/served_eval.py run --model qwen/qwen3-8b --label qwen3-8b_nothink \
      --docs 0-99 --suffix " /no_think" --max-cost 1.0
  python3 -I scripts/served_eval.py score results/round0/qwen3-8b_nothink/samples.jsonl
  # project deployment (fine-tuned model ref), output under results/round1/<label>/:
  python3 -I scripts/served_eval.py run --model ft-xxxxxxxx-qwen3-8b --label A_full \
      --docs 0-999 --out-root results/round1 --concurrency 16 --max-cost 3.0

`--suffix` is appended to the final user turn only (Qwen3 soft switch); the served provider
ignores chat_template_kwargs.enable_thinking, so this is the only way to get thinking off there.
Project deployments honour chat_template_kwargs.enable_thinking=false (always sent unless
--thinking). Deployment responses may carry no usage.cost, so --max-cost is then no guard:
watch the billing ledger instead.
The API key is read from .overmind/credentials.toml and never printed.
"""
import argparse
import concurrent.futures as cf
import json
import math
import pathlib
import re
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data/val_eval_5shot_chat.jsonl"
URL = "https://api.overmindlab.ai/api/v1/chat/completions"
UNTIL = ["Question:", "</s>", "<|im_end|>"]

# lm-eval gsm8k: filters + exact_match(ignore_case, regexes_to_ignore) -------------------
STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
FLEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
IGNORE = [",", r"\$", r"(?s).*#### ", r"\.$"]


def _pick(matches, group_select):
    if not matches:
        return "[invalid]"
    m = matches[group_select]
    if isinstance(m, tuple):
        m = [x for x in m if x]
        m = m[0] if m else "[invalid]"
    return m.strip()


def strict_extract(text):
    return _pick(STRICT.findall(text if isinstance(text, str) else ""), 0)


def flex_extract(text):
    return _pick(FLEX.findall(text if isinstance(text, str) else ""), -1)


def norm(s):
    for pat in IGNORE:
        s = re.sub(pat, "", s)
    return s.lower()


def correct(pred, gold):
    return norm(pred) == norm(gold)


# diagnostic only (NOT a competition metric): is the final number right, whatever the format?
# first `#### N` (also `#### **Answer: N**`), else last \boxed{N}, else last number; numeric compare.
def _num(s):
    try:
        return float(s.replace(",", "").replace("$", "").strip().rstrip("."))
    except ValueError:
        return None


def format_free_ok(text, gold):
    if not text:
        return False
    m = re.findall(r"#### \**\s*(?:Answer:\s*)?\$?(-?[0-9][0-9.,]*)", text)
    m = m[:1] or re.findall(r"\\boxed\{\s*\\?\$?(-?[0-9][0-9.,]*)", text)[-1:] \
        or re.findall(r"-?[0-9][0-9,]*\.?[0-9]*", text)[-1:]
    v, g = (_num(m[0]) if m else None), _num(gold)
    return v is not None and g is not None and abs(v - g) < 1e-6


# selftest: reproduce lm-eval's per-doc metrics from its own sample logs -------------------
SELFTEST_FILES = [
    "verification/r1-qwen3-4b/samples_gsm8k_val_2026-10-05T23-04-56.781655.jsonl",
    "results/round0/llama-3.2-3b-instruct_endpoint/meta-llama__llama-3.2-3b-instruct/"
    "samples_gsm8k_val_2026-10-08T12-54-42.592876.jsonl",
]


def selftest():
    ok = True
    for rel in SELFTEST_FILES:
        rows = [json.loads(line) for line in (ROOT / rel).open()]
        agree = total = mine = theirs = 0
        for r in rows:
            resp = r["resps"][0][0]
            pred = strict_extract(resp) if r["filter"] == "strict-match" else flex_extract(resp)
            if pred != r["filtered_resps"][0]:
                ok = False
            m = float(correct(pred, r["target"]))
            agree += m == r["exact_match"]
            total += 1
            if r["filter"] == "strict-match":
                mine += m
                theirs += r["exact_match"]
        n = total // 2
        print(f"{rel.split('/')[-1][:40]}: per-row agreement {agree}/{total}; "
              f"strict mine {mine / n:.4f} vs lm-eval {theirs / n:.4f} (n={n})")
        ok &= agree == total
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# runner ----------------------------------------------------------------------------------
def api_key():
    return tomllib.loads((ROOT / ".overmind/credentials.toml").read_text())["api-key"]


def call(key, body, attempts=8):
    data = json.dumps(body).encode()
    last = None
    for i in range(attempts):
        req = urllib.request.Request(URL, data=data, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                d = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            d = {"error": {"message": f"HTTP {e.code}", "body": e.read().decode("utf-8", "replace")[:300]}}
        except Exception as e:  # network hiccup
            d = {"error": {"message": f"{type(e).__name__}: {e}"}}
        if "error" not in d:
            return d
        last = d["error"]
        time.sleep(min(60, 5 * 2 ** i))
    return {"error": last}


def parse_docs(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def run(args):
    key = api_key()
    rows = [json.loads(line) for line in DATA.open()]
    outdir = ROOT / args.out_root / args.label
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "samples.jsonl"
    done = set()
    if path.exists():
        last = {}
        for line in path.open():
            r = json.loads(line)
            last[r["doc_id"]] = r  # last write wins, as in score()
        done = {d for d, r in last.items() if not (args.retry_errors and r.get("content") is None)}
    todo = [d for d in parse_docs(args.docs) if d not in done]
    spent = sum(json.loads(line).get("cost") or 0 for line in path.open()) if path.exists() else 0.0
    lock = threading.Lock()
    config = {"model": args.model, "suffix": args.suffix, "thinking": args.thinking,
              "max_tokens": args.max_tokens, "temperature": 0, "stop": UNTIL, "docs": args.docs,
              "dataset": str(DATA.relative_to(ROOT))}
    (outdir / "config.json").write_text(json.dumps(config, indent=1))
    print(f"{args.label}: {len(todo)} docs to run ({len(done)} already done), spent so far ${spent:.4f}")

    def one(doc_id):
        nonlocal spent
        with lock:
            if spent >= args.max_cost:
                return None
        msgs = [dict(m) for m in rows[doc_id]["input"]["messages"]]
        msgs[-1]["content"] += args.suffix
        body = {"model": args.model, "messages": msgs, "max_tokens": args.max_tokens,
                "temperature": 0, "stop": UNTIL,
                "chat_template_kwargs": {"enable_thinking": bool(args.thinking)}}
        d = call(key, body)
        gold = rows[doc_id]["expected_output"]
        rec = {"doc_id": doc_id, "gold": gold}
        if "error" in d:
            rec.update(error=d["error"], content=None)
        else:
            c = d["choices"][0]
            u = d.get("usage") or {}
            content = c["message"].get("content") or ""
            reasoning = c["message"].get("reasoning") or ""
            rec.update(
                content=content, finish_reason=c.get("finish_reason"), provider=d.get("provider"),
                prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                cost=u.get("cost"), reasoning_chars=len(reasoning),
                reasoning=reasoning if args.thinking else None)
        rec["strict_pred"] = strict_extract(rec.get("content"))
        rec["strict_ok"] = correct(rec["strict_pred"], gold)
        rec["flex_pred"] = flex_extract(rec.get("content"))
        rec["flex_ok"] = correct(rec["flex_pred"], gold)
        with lock:
            spent += rec.get("cost") or 0
            with path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with cf.ThreadPoolExecutor(args.concurrency) as ex:
        for i, rec in enumerate(ex.map(one, todo), 1):
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} done, spent ${spent:.4f}", flush=True)
    return score(path)


def score(path):
    recs = {}
    for line in pathlib.Path(path).open():
        r = json.loads(line)
        recs[r["doc_id"]] = r  # last write wins
    rs = [recs[k] for k in sorted(recs)]
    ok_rs = [r for r in rs if r.get("content") is not None]
    n = len(rs)
    acc = sum(r["strict_ok"] for r in rs) / n
    summary = {
        "n": n,
        "errors": n - len(ok_rs),
        "strict_match": round(acc, 4),
        "strict_stderr": round(math.sqrt(acc * (1 - acc) / (n - 1)), 4) if n > 1 else None,
        "flexible_extract": round(sum(r["flex_ok"] for r in rs) / n, 4),
        "format_free_diagnostic": round(sum(format_free_ok(r.get("content"), r["gold"]) for r in rs) / n, 4),
        "missing_hash_answer": sum(r["strict_pred"] == "[invalid]" for r in rs),
        "truncated_length": sum(r.get("finish_reason") == "length" for r in ok_rs),
        "think_tag_in_content": sum("<think>" in (r.get("content") or "") for r in ok_rs),
        "with_reasoning_tokens": sum((r.get("reasoning_tokens") or 0) > 0 for r in ok_rs),
        "mean_completion_tokens": round(sum(r.get("completion_tokens") or 0 for r in ok_rs) / max(1, len(ok_rs)), 1),
        "max_completion_tokens": max((r.get("completion_tokens") or 0 for r in ok_rs), default=0),
        "cost_usd": round(sum(r.get("cost") or 0 for r in rs), 6),
        "providers": sorted({r.get("provider") or "?" for r in ok_rs}),
        "doc_ids": f"{rs[0]['doc_id']}-{rs[-1]['doc_id']}" if rs else "",
    }
    (pathlib.Path(path).parent / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--docs", default="0-99")
    r.add_argument("--suffix", default="")
    r.add_argument("--thinking", action="store_true")
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--max-cost", type=float, required=True, help="stop starting new requests above this USD")
    r.add_argument("--retry-errors", action="store_true", help="re-run docs whose last record is an API error")
    r.add_argument("--out-root", default="results/round0", help="output root, relative to repo (label dir goes under it)")
    s = sub.add_parser("score")
    s.add_argument("path")
    a = p.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "run":
        return run(a)
    return score(a.path)


if __name__ == "__main__":
    sys.exit(main())
