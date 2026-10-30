"""`host-watch` - inspect and clear foreign model loads on a local backend."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from lm_optimizer.logging_config import get_logger, setup_logging
from lm_optimizer.services.foreign_watchdog import (
    BACKEND_LMSTUDIO,
    BACKEND_OLLAMA,
    ForeignModelWatchdog,
    build_watchdog,
)

logger = get_logger(__name__)
console = Console()


def _echo_probe_candidates(models) -> list:
    """Models ``connect(echo_probe=True)`` would legitimately load.

    The client picks the SMALLEST model, skipping embedding models, and loads it
    once to verify applied configs. Those loads are ours, not a foreign client.
    """
    cands = [m for m in models if "embed" not in (getattr(m, "id", "") or "").lower()]
    cands = [m for m in cands if getattr(m, "size_bytes", None) is not None]
    if not cands:
        return []
    smallest = min(cands, key=lambda m: getattr(m, "size_bytes", 0) or 0)
    return [smallest]


def host_watch(
    backend: str = typer.Option(
        "lm-studio", "--backend", "-b", help="Backend to inspect: lm-studio or ollama"
    ),
    expect: Optional[str] = typer.Option(
        None, "--expect", help="Model expected to be loaded (protected from unload)"
    ),
    protect: Optional[str] = typer.Option(
        None, "--protect", help="Comma-separated extra models to protect, never unloaded"
    ),
    unload_foreign: bool = typer.Option(
        False, "--unload-foreign", help="Actually unload foreign models (default: report only)"
    ),
    watch: int = typer.Option(
        1, "--watch", "-w", help="Repeat this many times (0 = forever), interval --interval"
    ),
    interval: float = typer.Option(
        5.0, "--interval", help="Seconds between --watch iterations"
    ),
    fail_on_foreign: bool = typer.Option(
        False, "--fail-on-foreign", help="Exit nonzero when a foreign model is found"
    ),
    json_out: Optional[Path] = typer.Option(
        None, "--json", help="Write findings to this JSON file"
    ),
):
    """Report (and optionally unload) models loaded by someone else.

    A local client can load a different model while an optimization run is
    benchmarking another one, which corrupts measurements and can abort the
    generation in flight. This command lists what the backend currently has
    loaded, flags anything outside --expect/--protect, and with
    --unload-foreign removes exactly those (never the expected models).
    """
    setup_logging()

    normalized = (backend or "").strip().lower()
    if normalized not in ("lm-studio", "lmstudio", "ollama"):
        console.print(f"[red]Invalid --backend: {backend} (expected lm-studio|ollama)[/red]")
        raise typer.Exit(2)
    normalized = BACKEND_OLLAMA if normalized == "ollama" else BACKEND_LMSTUDIO

    extra = [p.strip() for p in (protect or "").split(",") if p.strip()]

    async def _one_pass() -> dict:
        from lm_optimizer.backends.ollama.client import OllamaClient
        from lm_optimizer.config import config
        from lm_optimizer.services.lm_studio import LMStudioClient

        if normalized == BACKEND_OLLAMA:
            client = OllamaClient(base_url=config.ollama_base_url)
            await client.connect(echo_probe=False)
            wd: ForeignModelWatchdog = build_watchdog(
                client,
                normalized,
                model_under_test=expect,
                allow_unload=unload_foreign,
            )
            wd.protect(extra)
        else:
            client = LMStudioClient()
            # Load-free connect: this command must not trigger the echo probe,
            # which itself loads the smallest model.
            await client.connect(echo_probe=False)
            models = await client.list_models()
            expected = [expect] if expect else []
            expected += [getattr(m, "id", None) for m in _echo_probe_candidates(models)]
            expected += extra
            wd = ForeignModelWatchdog(
                client,
                normalized,
                expected_models=expected,
                allow_unload=unload_foreign,
            )
        try:
            report = await wd.check()
        finally:
            try:
                await client.close()
            except Exception:
                pass
        return {
            "checked_at": report.checked_at.isoformat(),
            "backend": report.backend,
            "inspected": report.inspected,
            "clean": report.clean,
            "expected": [m.model_id for m in report.expected],
            "foreign": [
                {
                    "model": f.model.model_id,
                    "instance_id": f.model.instance_id,
                    "reason": f.reason,
                }
                for f in report.foreign
            ],
            "unloaded": report.unloaded,
            "unload_failed": report.unload_failed,
            "error": report.error,
            "summary": report.summary(),
        }

    iterations = 0 if watch == 0 else max(1, watch)
    results = []
    n = 0
    try:
        while True:
            n += 1
            out = asyncio.run(_one_pass())
            results.append(out)
            console.print(f"[{out['checked_at']}] {out['summary']}")
            for f in out["foreign"]:
                console.print(
                    f"    [yellow]foreign[/yellow] {f['model']}"
                    + (f" (instance {f['instance_id']})" if f["instance_id"] else "")
                )
            if out["unloaded"]:
                console.print(f"    [green]unloaded[/green] {', '.join(out['unloaded'])}")
            if out["unload_failed"]:
                console.print(f"    [red]unload failed[/red] {', '.join(out['unload_failed'])}")
            if not out["inspected"]:
                console.print(f"    [red]{out['error']}[/red]")
            if iterations and n >= iterations:
                break
            if not iterations:
                asyncio.run(asyncio.sleep(interval))
                continue
            asyncio.run(asyncio.sleep(interval))
    except KeyboardInterrupt:
        console.print("\n[yellow]stopped[/yellow]")

    if json_out:
        import json

        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        console.print(f"[dim]findings -> {json_out}[/dim]")

    if fail_on_foreign and any(r["foreign"] for r in results):
        raise typer.Exit(1)