#!/usr/bin/env python3
"""Upload a local JSONL file to Overmind and land it as a dataset.

Why: `overmind dataset upload` (CLI 0.2.1) skips the inspect step the server now requires
("Inspect upload ... before creating a dataset"), so this does reserve -> chunks -> inspect ->
create itself. Landing auto-starts Overmind's dataset agent (billed, ~$0.6-0.9); check the
cells it makes and pin the one you train on.

Usage (repo root):
  python3 -I scripts/overmind_upload.py data/train_selfdistill_native_sft.jsonl --intent train

The API key is read from .overmind/credentials.toml and never printed.
"""
import argparse
import json
import pathlib
import sys
import time
import tomllib

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--intent", choices=["train", "eval"], required=True)
    a = p.parse_args()
    path = pathlib.Path(a.path)
    cred = tomllib.loads((ROOT / ".overmind/credentials.toml").read_text())
    base = cred["base-url"].rstrip("/")
    s = requests.Session()
    s.headers.update({"X-Api-Key": cred["api-key"], "Content-Type": "application/json"})

    def ok(r, what):
        if r.status_code >= 400:
            sys.exit(f"{what}: HTTP {r.status_code} {r.text[:500]}")
        return r.json()

    reserved = ok(s.post(f"{base}/api/uploads/", json={"filename": path.name}, timeout=60), "reserve")
    upload_id, chunk = reserved["upload_id"], int(reserved["chunk_bytes"])
    data = path.read_bytes()
    if len(data) > int(reserved["max_bytes"]):
        sys.exit(f"{path.name} is {len(data)} bytes; server limit {reserved['max_bytes']}")
    sent = 0
    while sent < len(data):
        # the server stores a chunk once however often it is sent, so a dropped connection is retried
        for attempt in range(6):
            try:
                r = s.put(f"{base}/api/uploads/{upload_id}/chunk/", params={"offset": sent},
                          data=data[sent:sent + chunk], headers={"Content-Type": "application/octet-stream"},
                          timeout=300)
                break
            except requests.RequestException as e:
                if attempt == 5:
                    sys.exit(f"chunk at offset {sent} failed: {type(e).__name__}")
                time.sleep(2 ** attempt)
        sent = ok(r, "chunk")["received"]
    inspected = ok(s.post(f"{base}/api/uploads/{upload_id}/inspect/",
                          json={"size": len(data), "filename": path.name}, timeout=300), "inspect")
    rows = sum(1 for _ in path.open())
    if inspected.get("bytes") != len(data) or inspected.get("rows") != rows:
        sys.exit(f"inspect mismatch: server {inspected}, local bytes={len(data)} rows={rows}")
    body = {"project": cred["project-id"], "name": path.name, "intent": a.intent,
            "source": {"upload_id": upload_id, "filename": path.name}}
    created = ok(s.post(f"{base}/api/datasets/", json=body, timeout=300), "create")
    print(json.dumps({"dataset": created["id"], "upload_id": upload_id, "bytes": len(data), "rows": rows,
                      "state": created.get("state")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
