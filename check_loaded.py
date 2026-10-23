import httpx
import asyncio

async def main():
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get('http://127.0.0.1:1234/api/v1/models', timeout=10)
        models = r.json().get('models', [])
        for m in models:
            print(f'{m["id"]}: loaded_instances_count={m.get("loaded_instances_count", 0)}')
            for inst in m.get('loaded_instances', []):
                print(f'  instance_id: {inst.get("instance_id")}')

asyncio.run(main())