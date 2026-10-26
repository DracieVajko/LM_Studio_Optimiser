"""Probe: which identifier does /api/v1/models/load accept?"""

import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:1234"


def post(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


for ident in (
    "recursivesi/flint",
    "recursivesi/flint@q4_k_m",
    "qwen/qwen3-coder-30b",
    "qwen/qwen3-coder-30b@q4_k_m",
    "prism-ml/bonsai-27b",
):
    status, resp = post("/api/v1/models/load", {"model": ident})
    print(f"{ident:40s} -> {status}")
    if status != 200:
        print(f"    {resp}")
    else:
        post("/api/v1/models/unload", {"model": ident})
        print("    loaded + unloaded")
