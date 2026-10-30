import re
import sys
import time
from pathlib import Path

import httpx

base = Path("lm_optimizer/ui")
js_files = list((base / "static/js").glob("*.js"))
templates = list((base / "templates").rglob("*.html"))
all_js = "\n".join(f.read_text(encoding="utf-8") for f in js_files)
all_html = "\n".join(f.read_text(encoding="utf-8") for f in templates)

# 1) IDs referenced via getElementById must appear somewhere (template or JS).
ids = set(re.findall(r"getElementById\('([\w-]+)'\)", all_js))
missing = sorted(i for i in ids if i not in all_html and f'id="{i}"' not in all_js and f"id=\\'{i}\\'" not in all_js)
print(f"getElementById refs: {len(ids)}, missing: {missing or 'none'}")

# 2) UI.xxx handlers used from inline onclick must exist on the UI object.
calls = set(re.findall(r"onclick=\"UI\.(\w+)\(", all_html + all_js))
ui_js = (base / "static/js/ui.js").read_text(encoding="utf-8")
undefined = sorted(c for c in calls if f"{c}(" not in ui_js and f"{c}:" not in ui_js and f"{c} =" not in ui_js)
print(f"inline UI handlers: {len(calls)}, undefined: {undefined or 'none'}")

# 3) Page + API smoke checks.
time.sleep(2)
pages = ["/", "/history", "/sandbox", "/experiments", "/lab", "/settings"]
api = ["/api/status", "/api/stats", "/api/models", "/api/runs?limit=1",
       "/api/health", "/api/settings", "/api/lab/quality-tasks", "/api/lab/duel-types",
       "/api/lab/sandbox-tasks"]
client = httpx.Client(base_url="http://127.0.0.1:8099", timeout=20, follow_redirects=False)
for path in pages + api:
    try:
        r = client.get(path)
        print(f"{r.status_code} {path}")
    except Exception as e:
        print(f"ERR {path}: {type(e).__name__}: {e}")

for asset in ["/static/css/app.css", "/static/js/app.js", "/static/js/sandbox.js",
              "/static/js/results.js", "/static/js/history.js", "/static/js/ui.js"]:
    try:
        r = client.get(asset)
        print(f"{r.status_code} {asset} ({len(r.content)} bytes)")
    except Exception as e:
        print(f"ERR {asset}: {e}")
