from lm_optimizer.database.repositories import run_repo
runs = run_repo.list_all()
for r in runs:
    m = getattr(r, 'model', None)
    mid = getattr(m, 'id', '?')
    st = getattr(getattr(r, 'status', None), 'value', getattr(r, 'status', '?'))
    best = getattr(r, 'best_config_id', None)
    n = len(getattr(r, 'configurations', []) or [])
    print(f'{mid} | {st} | best={bool(best)} | configs={len(getattr(r, "configurations", []) or [])}')