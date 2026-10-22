"""Sandbox runners: execute a sandbox task in an isolated workdir.

Implements the execution half of the master spec section 13:

- The model writes files into an isolated workdir; nothing outside it is
  touched.
- Network is off by default; only whitelisted executables may run.
- Every execution has a timeout and bounded output.
- The repair loop re-runs the task with evidence-only feedback.
- Artifacts are listed for retention; commands are sanitized before they
  are ever shown to the user.

The runner is intentionally conservative: it runs commands with the
platform's shell in a cwd that is the isolated workdir, with a timeout, and
records stdout/stderr/exit code. It does NOT sandbox the OS - it is a
workdir + allowlist + timeout policy, which is what a cross-platform
tool can honestly promise.
"""

from __future__ import annotations

import inspect
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)

from lm_optimizer.services.sandbox_lab import (
    RepairAttempt,
    SandboxPolicy,
    SandboxResult,
    SandboxTask,
    build_repair_feedback,
    repair_penalty,
    score_attempt,
)


@dataclass
class ExecutionResult:
    """Outcome of one command execution."""

    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    error: str = ""


def run_command(
    command: str,
    workdir: Path,
    policy: SandboxPolicy,
    timeout_s: float | None = None,
) -> ExecutionResult:
    """Run one command inside the isolated workdir under the policy.

    The command is NOT executed when the executable is not on the
    allowlist - the attempt fails with a clear reason instead.
    """
    started = time.perf_counter()
    if not policy.is_command_allowed(command):
        return ExecutionResult(
            ok=False,
            exit_code=None,
            stdout="",
            stderr="",
            duration_s=0.0,
            error=(
                f"command not allowed by the sandbox policy: "
                f"{policy.sanitize_command(command)}"
            ),
        )
    workdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # Network off by default: no proxy variables, and the policy flag is
    # documented (a true network namespace is out of scope for a
    # cross-platform helper).
    if not policy.network_enabled:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env.pop(key, None)
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(workdir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s or policy.timeout_s,
            errors="replace",
        )
        return ExecutionResult(
            ok=proc.returncode == 0,
            exit_code=proc.returncode,
            stdout=proc.stdout[-policy.max_output_bytes :],
            stderr=proc.stderr[-policy.max_output_bytes :],
            duration_s=time.perf_counter() - started,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            ok=False,
            exit_code=None,
            stdout="",
            stderr="",
            duration_s=time.perf_counter() - started,
            error=f"timeout after {timeout_s or policy.timeout_s:.0f}s",
        )
    except Exception as e:  # noqa: BLE001
        return ExecutionResult(
            ok=False,
            exit_code=None,
            stdout="",
            stderr="",
            duration_s=time.perf_counter() - started,
            error=f"{type(e).__name__}: {e}"[:200],
        )


def apply_retention(base_dir: Path, policy: SandboxPolicy) -> list[Path]:
    """Prune old run directories per the policy (spec §13.5).

    `artifact_retention` is "keep_last_N" (default keep_last_10). Returns the
    directories that were removed. The newest N runs are always kept.
    """
    removed: list[Path] = []
    if not base_dir.is_dir():
        return removed
    try:
        keep = int(str(policy.artifact_retention).split("_")[-1])
    except (TypeError, ValueError):
        keep = 10
    if keep < 0:
        keep = 10
    dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in dirs[keep:]:
        try:
            shutil.rmtree(stale)
            removed.append(stale)
        except OSError:
            continue
    return removed


def read_workdir_files(workdir: Path, limit: int = 50) -> dict[str, str]:
    """Read the text files the model produced (best effort, size limited)."""
    out: dict[str, str] = {}
    if not workdir.exists():
        return out
    count = 0
    for path in sorted(workdir.rglob("*")):
        if count >= limit:
            break
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > 512_000:
                continue
            # POSIX-style relative paths so reports are platform independent.
            rel = path.relative_to(workdir).as_posix()
            out[rel] = path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        count += 1
    return out


@dataclass
class SandboxRunner:
    """Executes a sandbox task with the repair loop."""

    task: SandboxTask
    policy: SandboxPolicy
    workdir: Path
    #: Command that runs the model-produced solution (e.g. a python script).
    command: str = ""
    #: Extra checks provided by the host (browser tests, blender validation).
    custom_checks: dict[str, tuple[bool, str]] = field(default_factory=dict)

    def run(self, generate) -> SandboxResult:
        """Synchronous wrapper around `run_async` (used by the tests)."""
        import asyncio

        return asyncio.run(self.run_async(generate))

    async def run_async(self, generate) -> SandboxResult:
        """Run the task with the repair loop.

        `generate(feedback: str) -> str` (or an async equivalent) is called
        for every attempt; the first call gets an empty feedback string,
        later ones get the evidence-only repair feedback. Async generators
        are awaited - the CLI talks to LM Studio through an async client.
        """
        result = SandboxResult(task_id=self.task.id)
        feedback = ""
        for attempt in range(self.task.max_repair_attempts + 1):
            started = time.perf_counter()
            try:
                produced = generate(feedback)
                if inspect.isawaitable(produced):
                    produced = await produced
            except Exception as e:  # noqa: BLE001
                result.error = f"{type(e).__name__}: {e}"[:200]
                break

            # The model returns the file contents; write them into the
            # isolated workdir.
            self._materialise(produced)

            execution = ExecutionResult(
                ok=True, exit_code=0, stdout="", stderr="", duration_s=0.0
            )
            if self.command:
                execution = run_command(
                    self.command, self.workdir, self.policy, self.task.timeout_s
                )

            files = read_workdir_files(self.workdir)
            # Web tasks get the static HTML/JS validation automatically
            # (spec 13.1 criteria achievable without a browser engine).
            custom = dict(self.custom_checks or {})
            if self.task.kind.value == "web":
                try:
                    from lm_optimizer.services.web_validator import (
                        validate_web_files,
                    )

                    for check in validate_web_files(files):
                        custom[check.name] = (check.passed, check.detail)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Web validation failed", error=str(e)[:160])
            score, failing, critical = score_attempt(
                self.task,
                files,
                stdout=execution.stdout,
                stderr=execution.stderr,
                exit_code=execution.exit_code,
                custom_results=custom or None,
            )
            passed = not failing and execution.ok
            record = RepairAttempt(
                attempt=attempt,
                output=produced[:4000],
                exit_code=execution.exit_code,
                passed=passed,
                score=score,
                error=execution.error or ("; ".join(failing)[:300] if failing else ""),
                duration_s=time.perf_counter() - started,
            )
            result.attempts.append(record)
            if attempt == 0:
                result.first_pass_success = passed
            if critical:
                result.critical_failures = critical
            if passed:
                result.final_success = True
                result.final_score = score
                result.penalized_score = score * repair_penalty(attempt)
                result.repair_count = attempt
                break
            if attempt == self.task.max_repair_attempts:
                result.final_score = score
                result.penalized_score = score * repair_penalty(attempt)
                result.repair_count = attempt
                break
            feedback = build_repair_feedback(
                self.task,
                attempt=attempt + 1,
                stdout=execution.stdout,
                stderr=execution.stderr,
                failing_checks=failing,
            )
        result.artifacts = sorted(read_workdir_files(self.workdir).keys())
        return result

    def _materialise(self, produced: str) -> None:
        """Write the model output into the isolated workdir."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        text = produced or ""
        # A model may answer with a single file; the task decides the name.
        primary = self.task.artifacts[0] if self.task.artifacts else "output.txt"
        if primary.endswith("/"):
            primary = primary.rstrip("/") + "/output.txt"
        target = self.workdir / primary
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def build_command_for(task: SandboxTask) -> str:
    """Default command that executes the produced artifact."""
    if task.kind.value == "blender":
        return "blender --background --python output.py"
    if task.kind.value == "web":
        return ""
    return ""
