"""Probe load identifiers with a long timeout (large models spill to RAM)."""

import httpx

BASE = "http://127.0.0.1:1234"

for ident in ("qwen/qwen3-coder-30b", "recursivesi/flint"):
    print(f"=== load by key: {ident} ===")
    with httpx.Client(timeout=420.0) as c:
        r = c.post(BASE + "/api/v1/models/load", json={"model": ident})
        print(f"  status={r.status_code} body={str(r.text)[:200]}")
        if r.status_code == 200:
            c.post(BASE + "/api/v1/models/unload", json={"model": ident}, timeout=180.0)
            print("  unloaded")
