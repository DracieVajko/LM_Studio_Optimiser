"""Sandbox Lab unit tests (P4, spec section 13)."""
import pytest

from lm_optimizer.services.sandbox_lab import (
    BLENDER_TASK,
    RESEARCH_PACKAGE,
    SANDBOX_TASKS,
    WEB_TASK,
    SandboxCheck,
    SandboxPolicy,
    SandboxResult,
    SandboxTask,
    SandboxTaskKind,
    build_repair_feedback,
    repair_penalty,
    score_attempt,
)

# --- task definitions ---

def test_all_three_sandbox_tasks_exist():
    assert set(SANDBOX_TASKS) == {"web_dashboard", "research_synthesis", "blender_skyblock"}
    kinds = {t.kind for t in SANDBOX_TASKS.values()}
    assert kinds == {SandboxTaskKind.WEB, SandboxTaskKind.RESEARCH, SandboxTaskKind.BLENDER}


def test_every_task_has_criteria_and_prompt():
    for task in SANDBOX_TASKS.values():
        assert task.prompt.strip()
        assert task.checks
        assert task.max_repair_attempts >= 1
        assert task.timeout_s > 0


def test_network_is_off_by_default():
    for task in SANDBOX_TASKS.values():
        assert task.network_allowed is False


def test_research_package_has_all_source_kinds():
    assert "primary_official" in RESEARCH_PACKAGE
    assert "technical_secondary" in RESEARCH_PACKAGE
    assert "outdated_source" in RESEARCH_PACKAGE
    assert "conflicting_source" in RESEARCH_PACKAGE
    assert "irrelevant_source" in RESEARCH_PACKAGE
    # The irrelevant source must be recognisable so it can be penalised.
    assert "cassette" in RESEARCH_PACKAGE["irrelevant_source"].lower()


def test_blender_task_checks_all_spec_elements():
    names = {c.name for c in BLENDER_TASK.checks}
    for required in (
        "has_camera",
        "has_light",
        "has_keyframes",
        "has_frame_range",
        "saves_blend",
        "renders",
    ):
        assert required in names


# --- scoring ---

def test_score_attempt_perfect_solution():
    files = {
        "index.html": (
            '<html><head><meta name="viewport" content="width=device-width">'
            "</head><body><table></table>"
            "<script>// dark theme toggle</script>"
            "<style>@media (max-width:600px){}</style></body></html>"
        )
    }
    score, failing, critical = score_attempt(WEB_TASK, files)
    assert score == pytest.approx(1.0)
    assert not failing
    assert not critical


def test_score_attempt_flags_external_resources():
    files = {
        "index.html": (
            '<html><head><meta name="viewport" content="x">'
            '<link rel="stylesheet" href="https://cdn.example.com/x.css">'
            "</head><body><div class=\"grid\"></div>"
            "<script src=\"https://cdn.example.com/x.js\"></script></body></html>"
        )
    }
    score, failing, critical = score_attempt(WEB_TASK, files)
    assert score < 1.0
    assert any("external" in f for f in failing)


def test_score_attempt_missing_critical_file():
    score, failing, critical = score_attempt(WEB_TASK, {"other.txt": "x"})
    assert critical
    assert any("index_html_exists" in c for c in critical)


def test_score_attempt_respects_custom_results():
    task = SandboxTask(
        id="t",
        kind=SandboxTaskKind.WEB,
        title="t",
        prompt="p",
        checks=[SandboxCheck("c1", "custom")],
    )
    score, failing, _ = score_attempt(task, {}, custom_results={"c1": (True, "ok")})
    assert score == pytest.approx(1.0)
    assert not failing


# --- repair loop ---

def test_repair_penalty_schedule():
    assert repair_penalty(0) == 1.0
    assert repair_penalty(1) == 0.9
    assert repair_penalty(2) == 0.8
    assert repair_penalty(5) == 0.8


def test_repair_feedback_contains_evidence_not_solution():
    feedback = build_repair_feedback(
        WEB_TASK,
        attempt=1,
        stdout="SyntaxError: unexpected token",
        stderr="line 3",
        failing_checks=["index_html_exists (missing=['index.html'])"],
        screenshot="shot.png",
    )
    assert "index_html_exists" in feedback
    assert "SyntaxError" in feedback
    assert "shot.png" in feedback
    assert "Remaining repair attempts: 1" in feedback


def test_repair_feedback_last_attempt_is_marked():
    feedback = build_repair_feedback(WEB_TASK, attempt=2, failing_checks=["x"])
    assert "LAST attempt" in feedback


def test_sandbox_result_serializes():
    result = SandboxResult(task_id="web_dashboard", first_pass_success=True, final_score=0.9)
    payload = result.as_dict()
    assert payload["first_pass_success"] is True
    assert "attempts" in payload


# --- security policy ---

def test_policy_blocks_unlisted_executables(tmp_path):
    policy = SandboxPolicy(workdir=tmp_path, allowed_executables=("blender",))
    assert policy.is_command_allowed("blender --background --python script.py")
    assert not policy.is_command_allowed("powershell -Command rm -rf C:\\")
    assert not policy.is_command_allowed("curl https://example.com")


def test_policy_with_empty_allowlist_blocks_everything(tmp_path):
    policy = SandboxPolicy(workdir=tmp_path)
    assert not policy.is_command_allowed("python script.py")


def test_policy_sanitizes_displayed_command(tmp_path):
    policy = SandboxPolicy(workdir=tmp_path, allowed_executables=("blender",))
    shown = policy.sanitize_command(
        '"C:\\Program Files\\Blender\\blender.exe" --background --python x.py'
    )
    assert shown.startswith("blender.exe")
    assert "Program Files" not in shown


def test_policy_sanitizes_posix_path_too(tmp_path):
    policy = SandboxPolicy(workdir=tmp_path, allowed_executables=("blender",))
    shown = policy.sanitize_command('"/usr/local/bin/blender" --background --python x.py')
    assert shown.startswith("blender")
    assert "/usr/local/bin" not in shown


def test_policy_accepts_both_path_styles(tmp_path):
    # Policy checks must not depend on the host that happens to run them.
    policy = SandboxPolicy(workdir=tmp_path, allowed_executables=("blender",))
    assert policy.is_command_allowed(
        '"C:\\Program Files\\Blender\\blender.exe" --background'
    )
    assert policy.is_command_allowed("/usr/local/bin/blender --background")


def test_policy_defaults_are_restrictive(tmp_path):
    policy = SandboxPolicy(workdir=tmp_path)
    assert policy.network_enabled is False
    assert policy.timeout_s > 0
    assert policy.memory_mb > 0
    assert policy.disk_mb > 0
