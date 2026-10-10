#!/usr/bin/env python3
"""Kall Overmind sine MCP-verktøy fra kommandolinjen (samme verktøy som mcp__overmind__* i Claude Code).

Bruk:
  python3 -I scripts/overmind_mcp.py list                      # list verktøy
  python3 -I scripts/overmind_mcp.py schema <verktøy>          # vis input-skjema
  python3 -I scripts/overmind_mcp.py call <verktøy> '<json>'   # kall verktøy med JSON-argumenter

API-nøkkelen leses fra .overmind/credentials.toml og skrives aldri ut.
"""
import json
import pathlib
import re
import sys
import tomllib
import urllib.error
import urllib.request

URL = "https://api.overmindlab.ai/api/mcp/"
ROOT = pathlib.Path(__file__).resolve().parent.parent
KEY = tomllib.loads((ROOT / ".overmind" / "credentials.toml").read_text())["api-key"]


def redact(text: str) -> str:
    return re.sub(r"ovr_[A-Za-z0-9_-]+", "ovr_***", text.replace(KEY, "***"))


class Client:
    def __init__(self):
        self.session = None
        self.next_id = 1

    def _post(self, payload, expect_reply=True):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "X-Api-Key": KEY,
        }
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                self.session = resp.headers.get("Mcp-Session-Id") or self.session
                body = resp.read().decode("utf-8", "replace")
                ctype = resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            raise SystemExit(f"HTTP {e.code}: {redact(e.read().decode('utf-8', 'replace'))[:800]}")
        if not expect_reply or not body.strip():
            return None
        if "text/event-stream" in ctype:
            msgs = [json.loads(line[5:].strip()) for line in body.splitlines() if line.startswith("data:") and line[5:].strip()]
            replies = [m for m in msgs if m.get("id") == payload.get("id")]
            return replies[-1] if replies else (msgs[-1] if msgs else None)
        return json.loads(body)

    def request(self, method, params=None):
        payload = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}}
        self.next_id += 1
        reply = self._post(payload)
        if reply is None:
            raise SystemExit(f"Tomt svar på {method}")
        if "error" in reply:
            raise SystemExit(f"MCP-feil på {method}: {redact(json.dumps(reply['error']))}")
        return reply["result"]

    def connect(self):
        self.request("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "overmind-mcp-cli", "version": "0.1"},
        })
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, expect_reply=False)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "schema", "call"):
        print(__doc__)
        sys.exit(2)
    client = Client()
    client.connect()
    cmd = sys.argv[1]
    if cmd in ("list", "schema"):
        tools = client.request("tools/list").get("tools", [])
        if cmd == "list":
            for t in tools:
                print(f"{t['name']}: {(t.get('description') or '').splitlines()[0][:150]}")
        else:
            match = [t for t in tools if t["name"] == sys.argv[2]]
            print(redact(json.dumps(match[0] if match else {"feil": "ukjent verktøy"}, indent=2, ensure_ascii=False)))
        return
    name = sys.argv[2]
    args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    result = client.request("tools/call", {"name": name, "arguments": args})
    out = []
    for part in result.get("content", []):
        out.append(part.get("text") if part.get("type") == "text" else json.dumps(part, ensure_ascii=False))
    if result.get("structuredContent") and not out:
        out.append(json.dumps(result["structuredContent"], indent=2, ensure_ascii=False))
    text = redact("\n".join(o for o in out if o))
    print(text)
    if result.get("isError"):
        sys.exit(1)


if __name__ == "__main__":
    main()
