"""Ollama Modelfile export + confirmed apply (no live calls here).

Export format is exactly one ``FROM <base>`` line followed by one
``PARAMETER <key> <value>`` per option (insertion order preserved)::

    FROM qwen3:8b
    PARAMETER num_ctx 8192
    PARAMETER temperature 0.7

``parse_modelfile`` is strict about structure (every non-blank,
non-comment line must be ``FROM`` or ``PARAMETER``) and about the value
types of known options; unknown keys pass through with lenient
int/float/bool coercion because Ollama silently ignores unknown
``options`` keys server-side. All structural problems are collected and
raised together as one ``ValueError`` naming every bad line
(``"line N: ..."``), so an edited file reports everything at once.

Apply (``apply_modelfile_text``) only ever targets a NEW tag — it lists
first and refuses when the tag already exists — and it never prompts
itself; confirmation (interactive prompt vs ``--yes``) stays the CLI's
job in ``lm_optimizer/cli/main.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lm_optimizer.backends.ollama.client import OllamaClient

#: Known integer-valued Ollama options (strictly validated on parse).
INT_KEYS = frozenset({"num_ctx", "num_batch", "num_thread", "num_gpu", "top_k"})

#: Known float-valued Ollama options (strictly validated on parse).
FLOAT_KEYS = frozenset({"temperature", "top_p", "min_p", "repeat_penalty"})

#: Known bool-valued Ollama options (``true``/``false`` on parse).
BOOL_KEYS = frozenset({"use_mmap"})


def _strip_inline_comment(value: str) -> str:
    """Remove a trailing ``# comment`` from a raw value."""
    return value.split("#", 1)[0].strip()


def coerce_param_value(key: str, raw: str) -> Any:
    """Coerce a raw ``PARAMETER``/``--param`` string to int/float/bool/str.

    Known keys are strictly typed (``ValueError`` on mismatch); unknown
    keys pass through with lenient coercion (bool > int > float > str).
    Raises plain ``ValueError`` without line info — callers add context.
    """
    text = _strip_inline_comment(raw.strip())
    if not text:
        raise ValueError(f"PARAMETER {key} needs a value")
    lowered = key.lower()
    if lowered in BOOL_KEYS:
        if text.lower() in ("true", "1", "yes", "on"):
            return True
        if text.lower() in ("false", "0", "no", "off"):
            return False
        raise ValueError(
            f"invalid value {raw.strip()!r} for PARAMETER {key} (expected true/false)"
        )
    if lowered in INT_KEYS:
        try:
            return int(text, 10)
        except ValueError:
            raise ValueError(
                f"invalid value {raw.strip()!r} for PARAMETER {key} (expected int)"
            ) from None
    if lowered in FLOAT_KEYS:
        try:
            return float(text)
        except ValueError:
            raise ValueError(
                f"invalid value {raw.strip()!r} for PARAMETER {key} (expected number)"
            ) from None
    # Unknown key: lenient pass-through (server ignores unknown keys anyway).
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return int(text, 10)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def parse_param_assignment(text: str) -> tuple[str, Any]:
    """Parse one ``--param KEY=VALUE`` CLI assignment (``ValueError`` if malformed)."""
    key, sep, raw = text.partition("=")
    key = key.strip()
    if not sep or not key or not raw.strip():
        raise ValueError(f"invalid --param {text!r} (expected KEY=VALUE)")
    return key, coerce_param_value(key, raw)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_modelfile(base: str, params: dict) -> str:
    """Render ``FROM`` + ``PARAMETER k v`` lines (trailing newline included)."""
    base = (base or "").strip()
    if not base:
        raise ValueError("render_modelfile: base model must be a non-empty tag")
    lines = [f"FROM {base}"]
    for key, value in dict(params or {}).items():
        lines.append(f"PARAMETER {key} {_format_value(value)}")
    return "\n".join(lines) + "\n"


def parse_modelfile(text: str) -> dict:
    """Parse Modelfile text to ``{"from": base, key: value, ...}``.

    Raises a single ``ValueError("line N: ...[; line M: ...]")`` covering
    every invalid line, or ``ValueError("line 1: missing FROM ...")``
    when no base line exists.
    """
    if not isinstance(text, str):
        raise ValueError("line 1: modelfile must be text")
    errors: list[str] = []
    result: dict = {}
    seen_from = False
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, rest = line.partition(" ")
        directive = head.upper()
        if directive == "FROM":
            base = _strip_inline_comment(rest)
            if not base:
                errors.append(f"line {lineno}: FROM needs a base model tag")
            elif seen_from:
                errors.append(
                    f"line {lineno}: duplicate FROM "
                    f"(base already {result['from']!r})"
                )
            else:
                result["from"] = base
                seen_from = True
        elif directive == "PARAMETER":
            parts = rest.split(None, 1)
            if len(parts) != 2 or not _strip_inline_comment(parts[1]):
                errors.append(
                    f"line {lineno}: PARAMETER needs KEY VALUE "
                    f"(got {raw_line.strip()!r})"
                )
                continue
            key, raw_value = parts[0], parts[1]
            try:
                result[key] = coerce_param_value(key, raw_value)
            except ValueError as e:
                errors.append(f"line {lineno}: {e}")
        else:
            errors.append(
                f"line {lineno}: unsupported directive {head!r} "
                "(expected FROM or PARAMETER)"
            )
    if errors:
        raise ValueError("; ".join(errors))
    if "from" not in result:
        raise ValueError("line 1: missing FROM <base> (first line must name the base model)")
    return result


def default_tag_for(base: str) -> str:
    """Default apply target for a base tag (``<name>:opt``)."""
    base = (base or "").strip()
    if not base:
        raise ValueError("default_tag_for: base model must be a non-empty tag")
    name = base.rsplit("/", 1)[-1].split(":")[0].strip() or base
    return f"{name}:opt"


async def ensure_tag_free(client: OllamaClient, tag: str) -> None:
    """Raise ``ValueError`` when ``tag`` already exists on the server.

    Compares case-insensitively (Ollama normalizes tags to lowercase).
    Server/transport errors propagate to the caller.
    """
    existing = [m.id for m in await client.list_models(force_refresh=True)]
    lowered = {e.lower() for e in existing}
    if tag.lower() in lowered:
        raise ValueError(
            f"refusing: target tag {tag!r} already exists "
            "(not overwriting; pick a new --as tag)"
        )


async def apply_modelfile_text(client: OllamaClient, text: str, tag: str) -> dict:
    """Validate ``text``, refuse taken tags, persist via ``POST /api/create``.

    Returns the decoded ``/api/create`` response. Raises ``ValueError``
    with ``"line N: ..."`` on invalid files and ``"already exists"``-style
    refusal when the tag is taken (re-checked immediately before create,
    so a tag created between prompt and apply is still refused).
    """
    parsed = parse_modelfile(text)
    await ensure_tag_free(client, tag)
    params = {k: v for k, v in parsed.items() if k != "from"}
    return await client.create(tag, from_=str(parsed["from"]), parameters=params)


__all__ = [
    "BOOL_KEYS",
    "FLOAT_KEYS",
    "INT_KEYS",
    "apply_modelfile_text",
    "coerce_param_value",
    "default_tag_for",
    "ensure_tag_free",
    "parse_modelfile",
    "parse_param_assignment",
    "render_modelfile",
]
