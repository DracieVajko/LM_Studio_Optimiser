"""Local `lms` CLI integration (read-only discovery + opt-in loads).

Findings (verified live, lms 1.3.3.0):
- `lms load --estimate-only <model>` prints memory estimates WITHOUT loading.
- `lms load <model> --gpu <off|max|0-1> -c <ctx> -y` loads with partial GPU
  offload (the gpu_ratio REST cannot set). The instance is then fully usable
  via REST (chat + unload by instance_id verified live).
- KV cache quantization, n_threads, try_mmap, keep-in-memory have NO CLI
  flags either (GUI-only, confirmed by absence in `lms load --help`).
"""

import re
import shutil
import subprocess

from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)

LMS_TIMEOUT_S = 600


def lms_available() -> bool:
    """Is the lms CLI on PATH?"""
    return shutil.which("lms") is not None


def lms_version() -> str:
    """CLI version string (best available version signal; REST has no version endpoint)."""
    if not lms_available():
        return "unknown (no lms CLI)"
    code, text = run_lms("--version", timeout=30)
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped:
            return stripped[:120]
    return f"unknown (rc={code})"


def runtime_survey() -> dict:
    """Parse `lms runtime survey` (backend engine id, GPUs, CPU, RAM)."""
    out: dict = {"engine": None, "gpus": [], "cpu": None, "ram_gib": None, "raw": ""}
    if not lms_available():
        out["raw"] = "lms CLI not found"
        return out
    code, text = run_lms("runtime", "survey", timeout=120)
    out["raw"] = text.strip()[-800:]
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Survey by "):
            out["engine"] = s[len("Survey by ") :][:160]
        elif s.startswith("CPU:"):
            out["cpu"] = s[4:].strip()[:120]
        elif s.startswith("RAM:"):
            m = re.search(r"([\d.]+)\s*GiB", s)
            if m:
                out["ram_gib"] = float(m.group(1))
        else:
            m = re.match(r"(.+?)\s+\(([^)]+)\)\s+([\d.]+)\s*GiB", s)
            if m and "GPU" in s.upper():
                out["gpus"].append(
                    {
                        "name": m.group(1).strip()[:120],
                        "mode": m.group(2).strip()[:60],
                        "vram_gib": float(m.group(3)),
                    }
                )
    out["returncode"] = code
    return out


def discover_cli_flags(commands: list[str] | None = None) -> dict:
    """Inventory CLI flags per command via `<cmd> --help` (runtime capability source).

    Returns {command: [flags]}. Never executes loads.
    """
    if not lms_available():
        return {}
    commands = commands or ["load", "ps", "ls", "runtime", "server", "log"]
    out: dict[str, list[str]] = {}
    for cmd in commands:
        _code, text = run_lms(cmd, "--help", timeout=30)
        flags = sorted(set(re.findall(r"--[a-zA-Z][a-zA-Z0-9-]*", text)))
        out[cmd] = flags
    return out


def run_lms(*args: str, timeout: int = LMS_TIMEOUT_S) -> tuple[int, str]:
    """Run lms with args, return (returncode, combined output). Never raises.

    Explicit UTF-8 with replacement: lms progress spinners emit bytes outside
    cp1252, which used to kill the reader thread with UnicodeDecodeError.
    """
    try:
        proc = subprocess.run(
            ["lms", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except Exception as e:
        return 127, f"lms execution failed: {e}"


def estimate(model_id: str) -> dict:
    """Memory estimate without loading. Returns {gpu_gib, total_gib, confidence, raw}."""
    return estimate_for(model_id)


def estimate_for(model_id: str, context: int | None = None, gpu: float | str | None = None) -> dict:
    """Memory estimate for a candidate (context/GPU offload) without loading.

    Extra flags are version-dependent: older `lms` may reject them, in which
    case this returns returncode != 0 and UNKNOWN numbers (never prune on
    parse failure; an explicit refusal MAY prune escalation, never proof).
    """
    out: dict = {
        "model": model_id,
        "gpu_gib": None,
        "total_gib": None,
        "confidence": None,
        "raw": "",
        "context": context,
        "gpu": gpu,
    }
    if not lms_available():
        out["raw"] = "lms CLI not found"
        return out
    args: list[str] = ["load", "--estimate-only", model_id]
    if context is not None:
        args += ["--context-length", str(context)]
    if gpu is not None:
        args += ["--gpu", str(gpu)]
    code, text = run_lms(*args)
    out["raw"] = text.strip()[-500:]
    gpu = re.search(r"Estimated GPU Memory:\s*([\d.]+)\s*GiB", text)
    total = re.search(r"Estimated Total Memory:\s*([\d.]+)\s*GiB", text)
    conf = re.search(r"Confidence:\s*(\w+)", text)
    if gpu:
        out["gpu_gib"] = float(gpu.group(1))
    if total:
        out["total_gib"] = float(total.group(1))
    if conf:
        out["confidence"] = conf.group(1)
    out["returncode"] = code
    return out


def cli_load(
    model_id: str, gpu_ratio: float | str | None = None, context: int | None = None
) -> tuple[str | None, str]:
    """Load via CLI (for gpu_ratio REST cannot set). Returns (identifier, error).

    Identifier defaults to the model key. Caller unloads via REST instance_id.
    """
    if not lms_available():
        return None, "lms CLI not found"
    args: list[str] = ["load", model_id, "-y"]
    if context is not None:
        args += ["--context-length", str(context)]
    if gpu_ratio is not None:
        args += ["--gpu", str(gpu_ratio)]
    code, text = run_lms(*args)
    if code != 0 or "Model loaded successfully" not in text:
        return None, f"lms load failed (rc={code}): {text.strip()[-300:]}"
    ident = re.search(r'identifier "([^"]+)"', text)
    identifier = ident.group(1) if ident else model_id
    logger.info("CLI load ok", model=model_id, identifier=identifier, gpu=gpu_ratio)
    return identifier, ""
