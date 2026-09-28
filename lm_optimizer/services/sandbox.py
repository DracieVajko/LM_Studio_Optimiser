"""Model-vs-model sandbox duels (text / html / scene).

Text duel: same prompt on model A then B with server-default load config
(empty LoadConfiguration — the user's LM Studio defaults apply). No tuning
parameters: identical neutral generation settings on both sides.

HTML/scene duel: each model generates one self-contained index.html into its
own pre-cleaned directory (A and B never share a folder). The scene kind
adds a small auto-rotating 3D-world instruction (three.js CDN).
"""

import asyncio
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from lm_optimizer.domain.models import LoadConfiguration
from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)

KINDS = ("text", "html", "scene")
DEFAULT_SANDBOX_ROOT = Path("results/sandbox")
DEFAULT_SIDE_TIMEOUT_S = 600
DEFAULT_MAX_FILE_CHARS = 200_000
TEXT_MAX_TOKENS = 1024
FILE_MAX_TOKENS = 8192

HTML_CONTRACT = (
    "Return ONLY one self-contained index.html file. "
    "No markdown fences, no explanations, no commentary — "
    "raw HTML starting with <!DOCTYPE html>."
)
SCENE_CONTRACT = (
    HTML_CONTRACT
    + " Use Three.js from CDN "
    "(https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js). "
    "Build a SMALL auto-rotating 3D environment (for example a tiny "
    "skyblock island) that rotates on its own. Nothing else."
)


@dataclass
class SandboxSide:
    ok: bool = False
    text: str = ""
    tok_s: float = 0.0
    elapsed_s: float = 0.0
    error: str | None = None
    file_path: str | None = None  # relative to sandbox root, forward slashes


@dataclass
class SandboxResult:
    job_id: str
    kind: str
    model_a: str
    model_b: str
    prompt: str
    side_a: SandboxSide = field(default_factory=SandboxSide)
    side_b: SandboxSide = field(default_factory=SandboxSide)
    faster: str | None = None  # "A" | "B" | "draw" | None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "prompt": self.prompt,
            "faster": self.faster,
            "side_a": {**self.side_a.__dict__},
            "side_b": {**self.side_b.__dict__},
        }


def clean_generated_html(raw: str) -> str:
    """Strip markdown fences / commentary wrappers; return raw HTML."""
    text = (raw or "").strip()
    fenced = re.match(r"^```(?:html)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    # Fences around a larger body: drop first fence line and trailing fence.
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _job_dir(sandbox_root: Path | str, job_id: str) -> Path:
    return Path(sandbox_root) / job_id


async def _run_side(
    client,
    model_id: str,
    full_prompt: str,
    max_output_tokens: int,
    timeout_s: float,
) -> SandboxSide:
    """Load with server defaults → one generation → unload. Never raises."""
    side = SandboxSide()
    start = time.perf_counter()
    try:
        res = await client.load_model(model_id, LoadConfiguration())
        if not getattr(res, "success", False):
            side.error = f"load failed: {getattr(res, 'error', '?')}"[:300]
            return side
        try:
            async def _gen():
                return await client.chat_completion(
                    model=model_id,
                    input_text=full_prompt,
                    temperature=0.7,
                    max_output_tokens=max_output_tokens,
                )

            try:
                response = await asyncio.wait_for(_gen(), timeout=timeout_s)
            except asyncio.TimeoutError:
                side.error = f"generation timed out after {timeout_s:.0f}s"
                return side
            choices = (response.get("choices") or [{}])[0]
            side.text = (choices.get("message", {}) or {}).get("content", "") or ""
            stats = response.get("_stats", {}) or {}
            side.tok_s = float(stats.get("tokens_per_second", 0) or 0)
            if not side.text.strip():
                side.error = "empty output"
                return side
            side.ok = True
        finally:
            try:
                await client.ensure_unloaded(model_id)
            except Exception:
                pass
    except Exception as e:
        side.error = f"{type(e).__name__}: {e}"[:300]
    finally:
        side.elapsed_s = round(time.perf_counter() - start, 1)
    return side


async def run_duel(
    client,
    model_a: str,
    model_b: str,
    prompt: str,
    kind: str = "text",
    max_output_tokens: int | None = None,
    timeout_s: float = DEFAULT_SIDE_TIMEOUT_S,
    max_file_chars: int = DEFAULT_MAX_FILE_CHARS,
    sandbox_root: Path | str = DEFAULT_SANDBOX_ROOT,
    job_id: str | None = None,
) -> SandboxResult:
    """Run model A, unload, then model B on the same prompt. Sequential only."""
    if kind not in KINDS:
        raise ValueError(f"Unknown sandbox kind {kind!r}, expected one of {KINDS}")
    if not (prompt or "").strip():
        raise ValueError("Sandbox prompt must not be empty")
    jid = job_id or uuid.uuid4().hex[:12]
    res = SandboxResult(job_id=jid, kind=kind, model_a=model_a, model_b=model_b, prompt=prompt)

    if kind == "text":
        full_prompt = prompt
        tokens = max_output_tokens or TEXT_MAX_TOKENS
    elif kind == "scene":
        full_prompt = SCENE_CONTRACT + "\n\nUser task: " + prompt
        tokens = max_output_tokens or FILE_MAX_TOKENS
    else:
        full_prompt = HTML_CONTRACT + "\n\nUser task: " + prompt
        tokens = max_output_tokens or FILE_MAX_TOKENS

    res.side_a = await _run_side(client, model_a, full_prompt, tokens, timeout_s)
    res.side_b = await _run_side(client, model_b, full_prompt, tokens, timeout_s)

    if kind in ("html", "scene"):
        root = _job_dir(sandbox_root, jid)
        # Fresh unique dir per run: inherently clean, sides isolated.
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        for tag, side in (("A", res.side_a), ("B", res.side_b)):
            if not side.ok:
                continue
            cleaned = clean_generated_html(side.text)
            if len(cleaned) > max_file_chars:
                side.ok = False
                side.error = (
                    f"output too large ({len(cleaned)} chars > {max_file_chars})"
                )
                side.text = ""
                continue
            if not cleaned:
                side.ok = False
                side.error = "empty output after cleaning"
                side.text = ""
                continue
            target = root / tag / "index.html"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(cleaned, encoding="utf-8")
            side.text = cleaned
            side.file_path = f"{jid}/{tag}/index.html"

    speeds = {t: s.tok_s for t, s in (("A", res.side_a), ("B", res.side_b)) if s.ok}
    if len(speeds) == 2:
        a, b = speeds["A"], speeds["B"]
        top = max(a, b)
        res.faster = "draw" if top <= 0 or abs(a - b) / top <= 0.05 else ("A" if a > b else "B")
    logger.info("Sandbox duel complete", job=jid, kind=kind, faster=res.faster)
    return res
