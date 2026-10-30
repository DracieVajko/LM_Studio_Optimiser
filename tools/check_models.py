import httpx
r = httpx.get('http://127.0.0.1:1234/api/v1/models', timeout=10)
for m in r.json().get('models', []):
    q = m.get('quantization', {})
    size = m.get('size_bytes', 0) / 1e9
    print(f'{m.get("key")} | {q.get("name", "?")} | {size:.2f} GB')