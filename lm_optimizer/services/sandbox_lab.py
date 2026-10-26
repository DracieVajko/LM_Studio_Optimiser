"""Sandbox Lab: web / research / blender tasks, repair loop, isolation.

Implements the master spec (LM_STUDIO_OPTIMISER_MASTER_SPEC_SK.md) section 13:

- 13.1 Web/HTML sandbox with precise acceptance criteria and automatic
  evaluation (build/runtime status, required files, console errors, network
  policy, functional tests, responsive screenshots, accessibility, visual
  overflow/overlap, maintainability).
- 13.2 Research sandbox: closed-source controlled package (primary,
  technical, outdated, conflicting and irrelevant sources) and open-web
  research. Penalize invented sources and non-existent citations.
- 13.3 Blender/3D sandbox: exact script exit code, .blend file, required
  objects, materials, camera, lights, keyframes, frame range, rendered
  frames, non-black frames, visible motion, composition.
- 13.4 Repair loop: max N attempts; the model receives stdout/stderr, failing
  tests, relevant screenshot and the remaining repair budget - never the
  solution. First-pass success and final success are reported separately.
  Penalty: first pass 100%, after one repair 90%, after two 80%.
- 13.5 Security: isolated workdir, network off by default, process timeout,
  CPU/RAM/disk limits, no host secrets, whitelist of executables, artifact
  retention, and no display of model-generated commands without sanitising.

The task DEFINITIONS and the evaluation LOGIC here are pure and offline
testable; the actual execution lives in the runner that talks to the
sandbox host.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SandboxTaskKind(str, Enum):
    WEB = "web"
    RESEARCH = "research"
    BLENDER = "blender"
    TEXT = "text"  # plain generation; scored only when checks are supplied


@dataclass
class SandboxCheck:
    """One acceptance criterion of a sandbox task.

    `negate=True` inverts the check: for a regex, a MATCH means FAIL
    (used for "must not contain an external script" style criteria).
    """

    name: str
    kind: str  # "files" | "regex" | "json" | "exit_code" | "custom"
    value: Any = None
    weight: float = 1.0
    critical: bool = False
    negate: bool = False


@dataclass
class SandboxTask:
    """A sandbox task definition (prompt + acceptance criteria)."""

    id: str
    kind: SandboxTaskKind
    title: str
    prompt: str
    checks: list[SandboxCheck] = field(default_factory=list)
    max_repair_attempts: int = 2
    timeout_s: float = 300.0
    network_allowed: bool = False
    allowed_executables: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()


# --- 13.1 Web task --------------------------------------------------------

WEB_TASK = SandboxTask(
    id="web_dashboard",
    kind=SandboxTaskKind.WEB,
    title="Responsive metrics dashboard",
    prompt=(
        "Build a self-contained HTML dashboard (single file, no external "
        "network calls) that shows four metrics, a filter, a model detail "
        "view, a light/dark theme toggle, and works on both mobile and "
        "desktop widths."
    ),
    checks=[
        SandboxCheck("index_html_exists", "files", ["index.html"], weight=1.0, critical=True),
        SandboxCheck("has_viewport_meta", "regex", r'name=["\']viewport["\']', weight=1.0),
        SandboxCheck("has_media_query", "regex", r"@media", weight=1.0),
        SandboxCheck("has_theme_toggle", "regex", r"(dark|theme)", weight=1.0),
        SandboxCheck("no_external_scripts", "regex", r"<script[^>]+src=[\"']https?://", weight=1.5, negate=True),
        SandboxCheck("no_external_styles", "regex", r"<link[^>]+href=[\"']https?://", weight=1.5, negate=True),
        SandboxCheck("has_table_or_grid", "regex", r"(<table|role=[\"']grid|class=[\"'][^\"']*grid)", weight=1.0),
    ],
    max_repair_attempts=2,
    timeout_s=300.0,
    network_allowed=False,
    allowed_executables=(),
    artifacts=("index.html",),
)


# --- 13.2 Research task ---------------------------------------------------

#: Closed-source research package: primary, technical, outdated, conflicting
#: and irrelevant sources. The model must handle the conflict and must not
#: invent citations.
RESEARCH_PACKAGE: dict[str, str] = {
    "primary_official": (
        "The LM Studio REST API v1 exposes POST /api/v1/models/load with the "
        "parameters context_length, flash_attention, offload_kv_cache_to_gpu, "
        "eval_batch_size, physical_batch_size, parallel, context_checkpoints, "
        "num_experts and speculative_draft_max_tokens. Unknown keys are "
        "rejected with HTTP 400 unrecognized_keys before the model is loaded."
    ),
    "technical_secondary": (
        "The llama.cpp server exposes speculative decoding through "
        "--spec-type draft-mtp and the MTP head shipped with Qwen3.8 models. "
        "On a dense 27B at Q4_K_M the acceptance rate of n=3 drafts was about "
        "73% and decode throughput rose from 31.2 to 40.3 tok/s on a V100."
    ),
    "outdated_source": (
        "Older guides state that a 27B Q4_K_M model needs 17.5 GB of VRAM "
        "and therefore cannot run on consumer cards at all. This guidance "
        "predates 3-bit quants and KV cache quantization."
    ),
    "conflicting_source": (
        "A community benchmark claims Q3_K_M is faster than Q4_K_M because the "
        "model is smaller, while another claims the opposite because Q3 "
        "requires more dequantization steps. Both were measured on Apple "
        "Silicon, where the Q3 kernel is known to be unusually slow."
    ),
    "irrelevant_source": (
        "A blog post about the history of the BASIC programming language "
        "describes how early microcomputers stored programs on cassette tapes."
    ),
}

RESEARCH_TASK = SandboxTask(
    id="research_synthesis",
    kind=SandboxTaskKind.RESEARCH,
    title="Source-grounded research synthesis",
    prompt=(
        "Using only the provided source documents, answer: which load "
        "parameters does the LM Studio REST API v1 accept, what is the "
        "measured effect of MTP speculative decoding on a dense 27B, and "
        "how should the conflicting Q3/Q4 speed claim be resolved? "
        "Cite every claim by source name. If the sources do not answer "
        "something, say so explicitly."
    ),
    checks=[
        SandboxCheck("cites_primary", "regex", r"primary_official", weight=2.0, critical=True),
        SandboxCheck("cites_technical", "regex", r"technical_secondary", weight=1.5),
        SandboxCheck("cites_conflicting", "regex", r"conflicting_source", weight=1.0),
        SandboxCheck("no_irrelevant_source", "regex", r"cassette", weight=2.0),
        SandboxCheck("mentions_unknown", "regex", r"(not (stated|provided|mentioned)|no (information|mention))", weight=1.0),
    ],
    max_repair_attempts=1,
    timeout_s=240.0,
    network_allowed=False,
    allowed_executables=(),
    artifacts=(),
)


# --- 13.3 Blender task ----------------------------------------------------

BLENDER_TASK = SandboxTask(
    id="blender_skyblock",
    kind=SandboxTaskKind.BLENDER,
    title="Minecraft-inspired Skyblock scene",
    prompt=(
        "Write a Blender Python script that builds a floating block island "
        "scene: a grass/dirt/stone block island, a tree, a chest, a block "
        "avatar, a sun light, and a camera that orbits the island. Set an "
        "exact frame range, render the frames and save the .blend file."
    ),
    checks=[
        SandboxCheck("script_defines_scene", "regex", r"(bpy\.ops\.mesh|def\s+\w+)", weight=2.0, critical=True),
        SandboxCheck("has_grass_material", "regex", r"(grass|meadow|green)", weight=1.5),
        SandboxCheck("has_tree", "regex", r"(tree|trunk|leaves|foliage)", weight=1.5),
        SandboxCheck("has_chest", "regex", r"chest", weight=1.0),
        SandboxCheck("has_camera", "regex", r"(Camera|camera)", weight=1.0),
        SandboxCheck("has_light", "regex", r"(Light|SUN|sun)", weight=1.0),
        SandboxCheck("has_keyframes", "regex", r"keyframe_insert", weight=1.5),
        SandboxCheck("has_frame_range", "regex", r"frame_(start|end)", weight=1.0),
        SandboxCheck("saves_blend", "regex", r"save_as_mainfile", weight=1.0, critical=True),
        SandboxCheck("renders", "regex", r"bpy\.ops\.render", weight=1.0),
    ],
    max_repair_attempts=2,
    timeout_s=600.0,
    network_allowed=False,
    allowed_executables=("blender",),
    artifacts=("scene.blend", "frames/"),
)


SANDBOX_TASKS: dict[str, SandboxTask] = {
    t.id: t for t in (WEB_TASK, RESEARCH_TASK, BLENDER_TASK)
}

#: Canonical alias used by the optimizer's post-run sandbox stage. Exactly
#: the same registry object - one source of truth for built-in tasks.
BUILTIN_TASKS: dict[str, SandboxTask] = SANDBOX_TASKS


def get_task_by_id(task_id: str) -> SandboxTask | None:
    """Resolve a built-in sandbox task by stable id (None when unknown)."""
    return SANDBOX_TASKS.get(task_id)


# --- Custom prompt schema + limits (spec §6.3) -----------------------------

CUSTOM_PROMPT_MAX_CHARS = 8000
CUSTOM_MAX_OUTPUT_TOKENS = 8192
CUSTOM_TIMEOUT_MIN_S = 30.0
CUSTOM_TIMEOUT_MAX_S = 3600.0
CUSTOM_TEMPERATURE_MIN = 0.0
CUSTOM_TEMPERATURE_MAX = 2.0

#: Display kind -> task kind. `text` stays a plain TEXT task.
_CUSTOM_KIND_MAP = {
    "text": SandboxTaskKind.TEXT,
    "html": SandboxTaskKind.WEB,
    "web": SandboxTaskKind.WEB,
    "scene": SandboxTaskKind.BLENDER,
    "blender": SandboxTaskKind.BLENDER,
    "research": SandboxTaskKind.RESEARCH,
}


def _stable_custom_id(prompt: str) -> str:
    """Deterministic id for a custom prompt (stable across processes)."""
    import hashlib

    return "custom_" + hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]


def validate_custom_prompt(spec: dict) -> dict:
    """Validate and normalise one custom sandbox prompt (spec §6.3).

    Enforces prompt/token/timeout/temperature limits and a valid kind.
    Raises ValueError with an actionable message on invalid input.
    """
    if not isinstance(spec, dict):
        raise ValueError("custom prompt must be an object")
    prompt = str(spec.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("custom prompt requires a non-empty 'prompt'")
    if len(prompt) > CUSTOM_PROMPT_MAX_CHARS:
        raise ValueError(
            f"custom prompt is {len(prompt)} chars; limit is {CUSTOM_PROMPT_MAX_CHARS}"
        )

    raw_kind = str(spec.get("kind") or "text").lower()
    kind = _CUSTOM_KIND_MAP.get(raw_kind)
    if kind is None:
        raise ValueError(
            f"unknown custom prompt kind {raw_kind!r}; "
            f"expected one of {', '.join(sorted(_CUSTOM_KIND_MAP))}"
        )

    try:
        max_tokens = int(spec.get("max_output_tokens", 2048))
    except (TypeError, ValueError):
        raise ValueError("max_output_tokens must be an integer") from None
    if not 1 <= max_tokens <= CUSTOM_MAX_OUTPUT_TOKENS:
        raise ValueError(
            f"max_output_tokens must be 1..{CUSTOM_MAX_OUTPUT_TOKENS} (got {max_tokens})"
        )

    try:
        timeout_s = float(spec.get("timeout_s", 300.0))
    except (TypeError, ValueError):
        raise ValueError("timeout_s must be a number") from None
    if not CUSTOM_TIMEOUT_MIN_S <= timeout_s <= CUSTOM_TIMEOUT_MAX_S:
        raise ValueError(
            f"timeout_s must be {CUSTOM_TIMEOUT_MIN_S:.0f}..{CUSTOM_TIMEOUT_MAX_S:.0f} "
            f"(got {timeout_s})"
        )

    try:
        temperature = float(spec.get("temperature", 0.2))
    except (TypeError, ValueError):
        raise ValueError("temperature must be a number") from None
    if not CUSTOM_TEMPERATURE_MIN <= temperature <= CUSTOM_TEMPERATURE_MAX:
        raise ValueError(
            f"temperature must be {CUSTOM_TEMPERATURE_MIN}..{CUSTOM_TEMPERATURE_MAX}"
        )

    checks_raw = spec.get("checks") or []
    if not isinstance(checks_raw, list):
        raise ValueError("checks must be a list")
    checks: list[SandboxCheck] = []
    for i, c in enumerate(checks_raw):
        if not isinstance(c, dict) or not c.get("name") or not c.get("kind"):
            raise ValueError(f"checks[{i}] requires 'name' and 'kind'")
        if c["kind"] not in ("files", "regex", "json", "exit_code"):
            raise ValueError(
                f"checks[{i}].kind must be files/regex/json/exit_code"
            )
        checks.append(
            SandboxCheck(
                name=str(c["name"]),
                kind=str(c["kind"]),
                value=c.get("value"),
                weight=float(c.get("weight", 1.0)),
                critical=bool(c.get("critical", False)),
                negate=bool(c.get("negate", False)),
            )
        )

    return {
        "id": str(spec.get("id") or _stable_custom_id(prompt)),
        "name": str(spec.get("name") or "Custom task")[:120],
        "prompt": prompt,
        "kind": kind.value,
        "task_kind": kind,
        "max_output_tokens": max_tokens,
        "temperature": temperature,
        "timeout_s": timeout_s,
        "checks": checks,
        # No expected criteria -> human review only, never a fabricated score.
        "manual_review_only": not checks,
    }


# --- 13.5 Security --------------------------------------------------------


def _basename_any(path_token: str) -> str:
    """Basename for both Windows and POSIX path styles.

    ``Path(...).name`` is host-dependent: on POSIX a Windows-style path is a
    single filename, so separators must be handled explicitly. This keeps
    policy checks and display sanitisation identical on every CI runner.
    """
    cleaned = str(path_token).strip().strip('"').strip("'")
    return re.split(r"[\\/]", cleaned)[-1]


@dataclass
class SandboxPolicy:
    """Isolation and resource policy (spec 13.5)."""

    workdir: Path
    network_enabled: bool = False
    timeout_s: float = 300.0
    max_output_bytes: int = 2_000_000
    cpu_seconds: int = 120
    memory_mb: int = 4096
    disk_mb: int = 512
    allowed_executables: tuple[str, ...] = ()
    artifact_retention: str = "keep_last_10"

    def is_command_allowed(self, command: str) -> bool:
        """Only whitelisted executables may run."""
        if not self.allowed_executables:
            return False
        try:
            parts = shlex.split(command, posix=False)
        except ValueError:
            return False
        if not parts:
            return False
        exe = _basename_any(parts[0]).lower()
        if exe.endswith(".exe"):
            exe = exe[: -len(".exe")]
        return exe in {e.lower().removesuffix(".exe") for e in self.allowed_executables}

    def sanitize_command(self, command: str) -> str:
        """Render a command for display without leaking host paths."""
        try:
            parts = shlex.split(command, posix=False)
        except ValueError:
            return "<unparsable command>"
        if not parts:
            return ""
        exe = _basename_any(parts[0])
        return " ".join([exe, *parts[1:]])


# --- 13.4 Repair loop -----------------------------------------------------


@dataclass
class RepairAttempt:
    """One attempt of a sandbox task."""

    attempt: int
    output: str = ""
    exit_code: int | None = None
    passed: bool = False
    score: float = 0.0
    error: str = ""
    duration_s: float = 0.0
    #: What the model is told for the NEXT attempt.
    feedback: str = ""


def build_repair_feedback(
    task: SandboxTask,
    attempt: int,
    stdout: str = "",
    stderr: str = "",
    failing_checks: list[str] | None = None,
    screenshot: str | None = None,
) -> str:
    """Feedback for the next repair attempt (spec 13.4).

    The model gets the failure EVIDENCE and the remaining budget - never the
    solution.
    """
    remaining = max(0, task.max_repair_attempts - attempt)
    lines = [
        f"The previous attempt FAILED validation ({attempt}/{task.max_repair_attempts}).",
        "",
        "Failing acceptance criteria:",
    ]
    for check in failing_checks or []:
        lines.append(f"  - {check}")
    if stdout.strip():
        lines += ["", "stdout (tail):", stdout.strip()[-1500:]]
    if stderr.strip():
        lines += ["", "stderr (tail):", stderr.strip()[-1500:]]
    if screenshot:
        lines += ["", f"screenshot: {screenshot}"]
    lines += [
        "",
        f"Remaining repair attempts: {remaining}.",
        "Fix the failing criteria. Do not rewrite unrelated parts.",
    ]
    if remaining == 0:
        lines.append("This is the LAST attempt - make it count.")
    return "\n".join(lines)


def score_attempt(
    task: SandboxTask,
    files: dict[str, str],
    stdout: str = "",
    stderr: str = "",
    exit_code: int | None = 0,
    custom_results: dict[str, tuple[bool, str]] | None = None,
) -> tuple[float, list[str], list[str]]:
    """Evaluate one attempt against the task's checks.

    Returns (score, failing_check_names, critical_failures).
    """
    total = 0.0
    weight_sum = 0.0
    failing: list[str] = []
    critical: list[str] = []
    for check in task.checks:
        ok = False
        detail = ""
        if custom_results and check.name in custom_results:
            ok, detail = custom_results[check.name]
        elif check.kind == "files":
            required = [str(f) for f in (check.value or [])]
            ok = all(f in files for f in required)
            detail = f"missing={[f for f in required if f not in files]}" if not ok else "ok"
        elif check.kind == "regex":
            pattern = str(check.value or "")
            haystack = "\n".join(files.values()) + "\n" + stdout + "\n" + stderr
            hit = re.search(pattern, haystack, re.IGNORECASE | re.DOTALL) is not None
            if check.negate:
                ok = not hit
                detail = "forbidden pattern present" if hit else "clean"
            else:
                ok = hit
                detail = "hit" if hit else "miss"
        elif check.kind == "exit_code":
            ok = exit_code == 0
            detail = f"exit={exit_code}"
        elif check.kind == "json":
            try:
                doc = json.loads("\n".join(files.values()))
                ok = all(k in doc for k in (check.value or []))
                detail = "ok" if ok else "missing keys"
            except Exception:
                ok = False
                detail = "invalid json"
        else:
            ok = False
            detail = f"unknown check kind {check.kind}"
        total += check.weight if ok else 0.0
        weight_sum += check.weight
        if not ok:
            failing.append(f"{check.name} ({detail})")
            if check.critical:
                critical.append(f"{check.name} ({detail})")
    score = (total / weight_sum) if weight_sum else 0.0
    return score, failing, critical


def repair_penalty(attempt_index: int) -> float:
    """Score multiplier per attempt (spec 13.4).

    First pass 100%, after one repair 90%, after two repairs 80%.
    """
    return {0: 1.0, 1: 0.9, 2: 0.8}.get(attempt_index, 0.8)


@dataclass
class SandboxResult:
    """Full sandbox run result (spec 13.1/13.4)."""

    task_id: str
    first_pass_success: bool = False
    final_success: bool = False
    repair_count: int = 0
    final_score: float = 0.0
    penalized_score: float = 0.0
    attempts: list[RepairAttempt] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    critical_failures: list[str] = field(default_factory=list)
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "first_pass_success": self.first_pass_success,
            "final_success": self.final_success,
            "repair_count": self.repair_count,
            "final_score": round(self.final_score, 4),
            "penalized_score": round(self.penalized_score, 4),
            "attempts": [
                {
                    "attempt": a.attempt,
                    "passed": a.passed,
                    "score": round(a.score, 4),
                    "error": a.error,
                    "duration_s": round(a.duration_s, 1),
                }
                for a in self.attempts
            ],
            "artifacts": list(self.artifacts),
            "critical_failures": list(self.critical_failures),
            "error": self.error,
        }
