"""Judge Service - orchestrates LLM-as-a-judge evaluations."""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from lm_optimizer.database.repositories import RunRepository, ConfigurationRepository
from lm_optimizer.services.model_judge import (
    ModelComparison,
    JudgeConfig,
    JudgeBackend,
    JudgeVerdict,
    create_judge,
)
from lm_optimizer.services.judge_prompt import PromptBuilder, JudgeRequest


@dataclass
class RunData:
    """Loaded run data for judge evaluation."""
    run_id: str
    model_id: str
    quality_threshold: float
    configs: list[dict]


class JudgeService:
    """Orchestrates judge evaluations: loads data, builds prompt, calls backend, saves verdict."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.run_repo = RunRepository()
        self.config_repo = ConfigurationRepository()

    async def _load_run_data(
        self,
        run_ids: list[str],
        config_ids: list[str] | None = None
    ) -> list[RunData]:
        """Load run data including configurations, metrics, and quality scores."""
        if not run_ids:
            raise ValueError("No run IDs provided")

        run_ids_quoted = ",".join(f"'{r}'" for r in run_ids)

        with closing(sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)) as conn, conn:
            conn.row_factory = sqlite3.Row

            # Get run metadata
            runs = conn.execute(
                f"SELECT id, model_id, quality_threshold FROM runs WHERE id IN ({run_ids_quoted})"
            ).fetchall()

            if not runs:
                raise ValueError("No runs found for provided IDs")

            results = []
            for run in runs:
                # Build config query
                config_query = """
                    SELECT id, run_id, config_json, metrics_json, quality_json,
                           stability_score, peak_vram_gb, peak_ram_gb, score, status, tested_at
                    FROM configurations
                    WHERE run_id = ? AND status IN ('passed', 'quality_failed', 'load_failed')
                """
                params = [run["id"]]

                if config_ids:
                    config_ids_quoted = ",".join(f"'{c}'" for c in config_ids)
                    config_query += f" AND id IN ({config_ids_quoted})"

                config_query += " ORDER BY score DESC NULLS LAST"

                configs = conn.execute(config_query, params).fetchall()

                if not configs:
                    continue  # Skip runs with no valid configs

                run_data = RunData(
                    run_id=run["id"],
                    model_id=run["model_id"],
                    quality_threshold=run["quality_threshold"],
                    configs=[]
                )

                for cfg in configs:
                    cfg_dict = {
                        "id": cfg["id"],
                        "run_id": cfg["run_id"],
                        "config_json": cfg["config_json"],
                        "metrics_json": cfg["metrics_json"],
                        "quality_json": cfg["quality_json"],
                        "stability_score": cfg["stability_score"],
                        "peak_vram_gb": cfg["peak_vram_gb"],
                        "peak_ram_gb": cfg["peak_ram_gb"],
                        "score": cfg["score"],
                        "status": cfg["status"],
                        "tested_at": cfg["tested_at"],
                    }
                    run_data.configs.append(cfg_dict)

                results.append(run_data)

            if not results:
                raise ValueError("No configurations to evaluate")

            return results

    def _select_backend(self, request: JudgeRequest):
        """Select and configure the appropriate judge backend."""
        config = JudgeConfig(
            backend=JudgeBackend(request.backend),
            model=request.judge_model,
            api_key=request.api_key,
            api_base=request.api_base,
            temperature=0.1,
            max_tokens=2048,
        )
        return create_judge(config)

    async def run_judge(self, request: JudgeRequest) -> JudgeVerdict:
        """Run full judge evaluation and persist verdict."""
        start_time = datetime.now()

        # 1. Load data
        run_data = await self._load_run_data(request.run_ids, request.config_ids)

        # 2. Build prompt
        prompt = PromptBuilder.build(run_data, request)

        # 3. Select backend
        judge = self._select_backend(request)

        try:
            # 4. Build comparison object for backend
            comparison = self._build_comparison(run_data, request)

            # 5. Evaluate
            verdict = await judge.evaluate(comparison)

            # 6. Add metadata
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            verdict.metadata = {
                "backend": request.backend,
                "judge_model": request.judge_model,
                "use_case": request.use_case,
                "priority": request.priority,
                "run_ids": request.run_ids,
                "configs_evaluated": sum(len(r.configs) for r in run_data),
                "prompt_tokens": PromptBuilder.estimate_tokens(prompt),
                "completion_tokens": 0,
                "latency_ms": latency_ms,
                "created_at": start_time.isoformat(),
            }

            # 7. Persist verdict
            self._save_verdict(request, verdict, latency_ms)

            return verdict

        finally:
            await judge.close()

    def _build_comparison(self, run_data: list[RunData], request: JudgeRequest) -> ModelComparison:
        """Convert run data to ModelComparison for backend."""
        # Flatten all configs
        all_configs = []
        for rd in run_data:
            for cfg in rd.configs:
                # Convert to ConfigurationResult-like dict
                all_configs.append({
                    "id": cfg["id"],
                    "model_id": rd.model_id,
                    "context_length": json.loads(cfg["config_json"]).get("context_length", 4096),
                    "score": cfg["score"],
                    "stability_score": cfg["stability_score"],
                    "peak_vram_gb": cfg["peak_vram_gb"],
                    "metrics_json": cfg["metrics_json"],
                    "quality_json": cfg["quality_json"],
                    "config_json": cfg["config_json"],
                })

        # Determine hardware constraints
        hardware = {
            "vram_gb": 6,
            "ram_gb": 32,
        }

        return ModelComparison(
            run=None,
            configs=all_configs,
            use_case=request.use_case,
            hardware_constraints=hardware,
            priority=request.priority,
        )

    def _save_verdict(self, request: JudgeRequest, verdict: JudgeVerdict, latency_ms: int) -> None:
        """Save verdict to database."""
        verdict_id = str(uuid4())

        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                """
                INSERT INTO judge_verdicts (
                    id, run_ids_json, config_ids_json, use_case, custom_prompt,
                    priority, backend, judge_model, verdict_json,
                    prompt_tokens, completion_tokens, latency_ms, status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    verdict_id,
                    json.dumps(request.run_ids),
                    json.dumps(request.config_ids) if request.config_ids else None,
                    request.use_case,
                    request.custom_prompt,
                    request.priority,
                    request.backend,
                    request.judge_model,
                    json.dumps(verdict.to_dict()),
                    verdict.metadata.get("prompt_tokens", 0),
                    verdict.metadata.get("completion_tokens", 0),
                    latency_ms,
                    "completed",
                    None,
                ),
            )