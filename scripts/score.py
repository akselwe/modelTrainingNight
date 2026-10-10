"""Compute total parameter count and the hackathon score.

Usage: python score.py <model_dir> [<lm_eval_results_dir>]
Score = strict-match accuracy (%) - total params (billions). Unrounded.
Counts ALL parameters in the checkpoint (base model incl. embeddings; merged LoRA).
If the checkpoint has a vision tower, it is reported separately - ask the organizers
whether it counts (Qwen 3.5 small models may ship one).
"""
import glob, json, os, sys
from safetensors import safe_open

model_dir = sys.argv[1]
total, vision = 0, 0
for f in glob.glob(os.path.join(model_dir, "**/*.safetensors"), recursive=True):
    with safe_open(f, "pt") as st:
        for k in st.keys():
            n = 1
            for d in st.get_slice(k).get_shape():
                n *= d
            total += n
            if any(t in k.lower() for t in ("visual", "vision", "vit", "image")):
                vision += n
print(f"total params: {total:,} ({total/1e9:.4f} B)")
if vision:
    print(f"  of which vision-tower params: {vision:,} ({vision/1e9:.4f} B)  <- confirm with organizers")
    print(f"  text-only params: {(total-vision)/1e9:.4f} B")

if len(sys.argv) > 2:
    files = glob.glob(os.path.join(sys.argv[2], "**/results_*.json"), recursive=True)
    if files:
        res = json.load(open(sorted(files)[-1]))["results"]["gsm8k"]
        acc = res["exact_match,strict-match"] * 100
        flex = res.get("exact_match,flexible-extract", float("nan")) * 100
        print(f"strict-match accuracy: {acc:.4f}%   (flexible-extract: {flex:.2f}%)")
        print(f"SCORE (total params):  {acc - total/1e9:.4f}")
        if vision:
            print(f"SCORE (text-only):     {acc - (total-vision)/1e9:.4f}")
