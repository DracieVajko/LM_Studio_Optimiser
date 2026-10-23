import httpx

r = httpx.get('http://127.0.0.1:1234/api/v1/models', timeout=10)
models = r.json().get('models', [])

for m in models:
    if m.get('id'):
        print(f'ID: {m["id"]} | Key: {m["key"]} | Display: {m.get("display_name")}')