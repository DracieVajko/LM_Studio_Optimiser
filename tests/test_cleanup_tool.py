"""tools/cleanup.py: archive-then-delete run data, logs, caches (offline).

All filesystem work happens in tmp_path; the production data/ directory is
never touched. lock_alive() is monkeypatched either way.
"""
import sqlite3

import pytest

from tools import cleanup as cleanup_mod


RUN_DB = "optimizer.db"


@pytest.fixture()
def fake_repo(tmp_path):
    """A minimal repo replica: data dir with DB, logs, reports, caches."""
    root = tmp_path / "repo"
    data = root / "data"
    logs = data / "logs"
    reports = data / "reports"
    checkpoints = data / "checkpoints"
    for d in (logs, reports, checkpoints):
        d.mkdir(parents=True)
    (logs / "run.log").write_text("log", encoding="utf-8")
    (logs / "sub").mkdir()
    (logs / "sub" / "a.log").write_text("log", encoding="utf-8")
    (reports / "report.md").write_text("# report", encoding="utf-8")
    (checkpoints / "c.json").write_text("{}", encoding="utf-8")
    (root / ".pytest_cache").mkdir()
    (root / ".pytest_cache" / "v").write_text("x", encoding="utf-8")
    (root / "src" / "__pycache__").mkdir(parents=True)
    (root / "src" / "__pycache__" / "m.pyc").write_bytes(b"\x00")
    (root / ".venv" / "lib" / "__pycache__").mkdir(parents=True)
    (root / ".venv" / "lib" / "__pycache__" / "v.pyc").write_bytes(b"\x00")

    conn = sqlite3.connect(str(data / RUN_DB))
    tables = {
        "runs": "id INTEGER, model_id TEXT, status TEXT",
        "configurations": "id INTEGER, run_id INTEGER, status TEXT",
        "benchmarks": "id INTEGER, config_id INTEGER",
        "quality_results": "id INTEGER, config_id INTEGER",
        "capability_snapshots": "id INTEGER, run_id INTEGER",
        "judge_verdicts": "id INTEGER, run_ids_json TEXT",
        "duels": "id INTEGER",
        "hardware_snapshots": "id INTEGER",
        "models": "id INTEGER, name TEXT",
        "presets": "id INTEGER, model_id INTEGER, config_json TEXT",
        "settings": "key TEXT, value TEXT",
        "migrations": "id INTEGER, name TEXT",
    }
    for name, cols in tables.items():
        conn.execute(f"CREATE TABLE {name} ({cols})")
    conn.execute("INSERT INTO runs VALUES (1, 'm', 'done')")
    conn.execute("INSERT INTO runs VALUES (2, 'm', 'interrupted')")
    conn.execute("INSERT INTO configurations VALUES (1, 1, 'passed')")
    conn.execute("INSERT INTO benchmarks VALUES (1, 1)")
    conn.execute("INSERT INTO capability_snapshots VALUES (1, 1)")
    conn.execute("INSERT INTO hardware_snapshots VALUES (1)")
    conn.execute("INSERT INTO judge_verdicts VALUES (1, '[]')")
    conn.execute("INSERT INTO duels VALUES (1)")
    conn.execute("INSERT INTO models VALUES (1, 'swift-qwen3.8-27b')")
    conn.execute("INSERT INTO presets VALUES (1, 1, '{}')")
    conn.execute("INSERT INTO settings VALUES ('k', 'v')")
    conn.execute("INSERT INTO migrations VALUES (1, 'init')")
    conn.commit()
    conn.close()
    return root


def _counts(data_dir):
    conn = sqlite3.connect(str(data_dir / RUN_DB))
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        return {t: conn.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
                for t in tables}
    finally:
        conn.close()


def _run(root, data, *extra, monkeypatch):
    output = []

    class _Out:
        def __call__(self, *a):
            pass

    monkeypatch.setattr(cleanup_mod, "lock_alive", lambda: False)
    code = cleanup_mod.main([
        "--repo-root", str(root), "--data-dir", str(data), *extra,
    ])
    return code


def test_dry_run_changes_nothing(fake_repo, monkeypatch):
    data = fake_repo / "data"
    code = _run(fake_repo, data, "--dry-run", monkeypatch=monkeypatch)
    assert code == 0
    assert not (data / "archive").exists()
    counts = _counts(data)
    assert counts["runs"] == 2 and counts["configurations"] == 1
    assert (data / "logs" / "run.log").exists()
    assert (fake_repo / ".pytest_cache").exists()


def test_full_cleanup_archives_then_deletes(fake_repo, monkeypatch):
    root = fake_repo
    data = root / "data"
    code = _run(root, data, "--yes", monkeypatch=monkeypatch)
    assert code == 0

    # Archive snapshot exists and still holds the deleted rows.
    archives = list((data / "archive").iterdir())
    assert len(archives) == 1
    snap = archives[0] / RUN_DB
    assert snap.exists()
    snap_counts = _counts(archives[0])
    assert snap_counts["runs"] == 2 and snap_counts["models"] == 1

    # Live DB: run/benchmark rows gone; models/presets/settings kept.
    live = _counts(data)
    assert live["runs"] == 0 and live["configurations"] == 0
    assert live["benchmarks"] == 0 and live["judge_verdicts"] == 0
    assert live["duels"] == 0 and live["hardware_snapshots"] == 0
    assert live["models"] == 1 and live["presets"] == 1
    assert live["settings"] == 1 and live["migrations"] == 1

    # reports/checkpoints moved into the archive; logs emptied in place.
    assert (archives[0] / "reports" / "report.md").exists()
    assert (archives[0] / "checkpoints" / "c.json").exists()
    assert not (data / "reports" / "report.md").exists()
    assert list((data / "logs").iterdir()) == []
    # Caches removed; venv untouched.
    assert not (root / ".pytest_cache").exists()
    assert not (root / "src" / "__pycache__").exists()
    assert (root / ".venv" / "lib" / "__pycache__").exists()


def test_run_ids_archive_roll_dir_is_timestamped(fake_repo, monkeypatch):
    data = fake_repo / "data"
    _run(fake_repo, data, "--yes", monkeypatch=monkeypatch)
    name = next((data / "archive").iterdir()).name
    assert name.startswith("cleanup_") and len(name) == len("cleanup_YYYYmmdd_HHMMSS")


def test_refused_while_lock_alive(fake_repo, monkeypatch):
    data = fake_repo / "data"
    monkeypatch.setattr(cleanup_mod, "lock_alive", lambda: True)
    code = cleanup_mod.main([
        "--repo-root", str(fake_repo), "--data-dir", str(data), "--yes",
    ])
    assert code == 1
    assert _counts(data)["runs"] == 2  # nothing happened
    assert not (data / "archive").exists()


def test_logs_only_skips_db(fake_repo, monkeypatch):
    data = fake_repo / "data"
    code = _run(fake_repo, data, "--yes", "--logs-only",
                monkeypatch=monkeypatch)
    assert code == 0
    assert _counts(data)["runs"] == 2  # DB untouched
    assert (data / "reports" / "report.md").exists()  # not archived/moved
    assert list((data / "logs").iterdir()) == []


def test_missing_db_falls_back_to_logs_only(fake_repo, monkeypatch):
    (fake_repo / "data" / RUN_DB).unlink()
    code = _run(fake_repo, fake_repo / "data", "--yes",
                monkeypatch=monkeypatch)
    assert code == 0
    assert list((fake_repo / "data" / "logs").iterdir()) == []
