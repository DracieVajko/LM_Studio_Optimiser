"""Sandbox runner unit tests (P4 execution half)."""
import sys
from pathlib import Path

import pytest

from lm_optimizer.services.sandbox_lab import (
    SandboxPolicy,
    SandboxTask,
    SandboxTaskKind,
)
from lm_optimizer.services.sandbox_runner import (
    SandboxRunner,
    read_workdir_files,
    run_command,
)


def _task(**kw):
    base = dict(
        id="t",
        kind=SandboxTaskKind.WEB,
        title="t",
        prompt="p",
        checks=[],
        max_repair_attempts=2,
        timeout_s=10,
        artifacts=("index.html",),
    )
    base.update(kw)
    return SandboxTask(**base)


def test_run_command_blocks_non_allowlisted(tmp_path):
    policy = SandboxPolicy(workdir=tmp_path, allowed_executables=("python",))
    result = run_command("cmd /c echo hi", tmp_path, policy)
    assert not result.ok
    assert "not allowed" in result.error
    # The error must not leak the full host path.
    assert str(tmp_path) not in result.error


def test_run_command_runs_allowlisted(tmp_path):
    policy = SandboxPolicy(workdir=tmp_path, allowed_executables=("python",))
    result = run_command(
        f'{sys.executable} -c "print(123)"', tmp_path, policy, timeout_s=30
    )
    assert result.ok
    assert "123" in result.stdout


def test_run_command_respects_timeout(tmp_path):
    policy = SandboxPolicy(workdir=tmp_path, allowed_executables=("python",))
    result = run_command(
        f'{sys.executable} -c "import time; time.sleep(5)"',
        tmp_path,
        policy,
        timeout_s=1,
    )
    assert not result.ok
    assert "timeout" in result.error


def test_run_command_captures_nonzero_exit(tmp_path):
    policy = SandboxPolicy(workdir=tmp_path, allowed_executables=("python",))
    result = run_command(
        f'{sys.executable} -c "import sys; sys.exit(3)"', tmp_path, policy, timeout_s=30
    )
    assert not result.ok
    assert result.exit_code == 3


def test_read_workdir_files(tmp_path):
    (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("x", encoding="utf-8")
    files = read_workdir_files(tmp_path)
    assert "index.html" in files
    assert "sub/a.txt" in files


def test_runner_scores_and_repairs(tmp_path):
    from lm_optimizer.services.sandbox_lab import SandboxCheck

    task = _task(
        checks=[
            SandboxCheck("has_html", "regex", r"<html", weight=1.0),
            SandboxCheck("has_title", "regex", r"<title", weight=1.0),
        ]
    )
    policy = SandboxPolicy(workdir=tmp_path / "wd", allowed_executables=())
    runner = SandboxRunner(task=task, policy=policy, workdir=tmp_path / "wd")

    calls = []

    def generate(feedback: str) -> str:
        calls.append(feedback)
        if len(calls) == 1:
            return "<html>no title</html>"
        return "<html><title>ok</title></html>"

    result = runner.run(generate)
    assert result.first_pass_success is False
    assert result.final_success is True
    assert result.repair_count == 1
    # The second attempt must receive EVIDENCE, not a solution.
    assert calls[1]
    assert "has_title" in calls[1]
    assert result.penalized_score < result.final_score


def test_runner_reports_failure_after_budget(tmp_path):
    from lm_optimizer.services.sandbox_lab import SandboxCheck

    task = _task(
        max_repair_attempts=1,
        checks=[SandboxCheck("impossible", "regex", r"NEVER_MATCHES_XYZ", weight=1.0)],
    )
    policy = SandboxPolicy(workdir=tmp_path / "wd")
    runner = SandboxRunner(task=task, policy=policy, workdir=tmp_path / "wd")

    def generate(feedback: str) -> str:
        return "<html></html>"

    result = runner.run(generate)
    assert result.final_success is False
    assert len(result.attempts) == 2  # first try + one repair
    assert result.repair_count == 1


def test_runner_materialises_into_workdir(tmp_path):
    task = _task()
    workdir = tmp_path / "wd"
    policy = SandboxPolicy(workdir=workdir)
    runner = SandboxRunner(task=task, policy=policy, workdir=workdir)
    runner.run(lambda feedback: "<html>done</html>")
    assert (workdir / "index.html").read_text(encoding="utf-8") == "<html>done</html>"
