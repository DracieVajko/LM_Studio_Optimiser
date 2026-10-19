"""Prompt Builder for LLM Judge - constructs prompts from run data."""

import json
from dataclasses import dataclass
from typing import Any

from lm_optimizer.services.model_judge import JudgeBackend


@dataclass
class JudgeRequest:
    """Request parameters for judge evaluation."""
    run_ids: list[str]
    config_ids: list[str] | None = None
    use_case: str = "general"
    custom_prompt: str | None = None
    priority: str = "balanced"
    backend: str = "heuristic"
    judge_model: str = "heuristic"
    api_key: str | None = None
    api_base: str | None = None
    top_n_detail: int = 3
    include_reasoning_matrix: bool = True
    fastest_context_preset: bool = False


class PromptBuilder:
    """Builds LLM prompts for judge evaluation from run/config data."""

    # Use-case specific guidance
    USE_CASE_GUIDANCE = {
        "coding": "Prioritize code correctness, syntax validity, and problem-solving ability. Weight coding_task heavily.",
        "chat": "Prioritize natural conversation flow, instruction following, and TTFT. Weight short_instruction and medium_reasoning.",
        "reasoning": "Prioritize step-by-step logical reasoning and long-context comprehension. Weight medium_reasoning and long_context.",
        "structured": "Prioritize exact format compliance (JSON, XML, etc.). Weight structured_output heavily.",
        "general": "Balanced evaluation across all capabilities.",
    }

    # Hardware info for context
    HARDWARE_DESCRIPTION = "RTX 3060 Laptop 6GB VRAM, 32GB RAM"

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation: ~4 chars per token for English."""
        return len(text) // 4

    @classmethod
    def build(cls, run_data: list, request: JudgeRequest) -> str:
        """Build complete prompt for judge evaluation."""
        # Flatten all configs
        all_configs = []
        for rd in run_data:
            for cfg in rd.configs:
                cfg_copy = dict(cfg)
                cfg_copy["model_id"] = rd.model_id
                all_configs.append(cfg_copy)

        # Apply fastest_context_preset filter
        if request.fastest_context_preset:
            all_configs = cls._filter_fastest_at_max_context(all_configs)

        # Sort by score descending
        all_configs.sort(key=lambda c: c.get("score") or 0, reverse=True)

        # Build sections
        system_prompt = cls._build_system_prompt(request)
        hardware_section = f"HARDWARE: {cls.HARDWARE_DESCRIPTION}\n"
        summary_table = cls._build_summary_table(all_configs)
        top_n_detail = cls._build_top_n_detail(all_configs, request.top_n_detail)
        reasoning_matrix = cls._build_reasoning_matrix(all_configs) if request.include_reasoning_matrix else ""
        task_instruction = cls._build_task_instruction(request)

        # Combine
        prompt_parts = [
            system_prompt,
            hardware_section,
            summary_table,
            top_n_detail,
            reasoning_matrix,
            task_instruction,
        ]

        prompt = "\n".join(p for p in prompt_parts if p)

        # Auto-reduce top_n_detail if prompt too long (>8000 tokens estimated)
        while cls.estimate_tokens(prompt) > 8000 and request.top_n_detail > 1:
            request.top_n_detail -= 1
            top_n_detail = cls._build_top_n_detail(all_configs, request.top_n_detail)
            prompt_parts = [
                system_prompt,
                hardware_section,
                summary_table,
                top_n_detail,
                reasoning_matrix,
                task_instruction,
            ]
            prompt = "\n".join(p for p in prompt_parts if p)

        return prompt

    @classmethod
    def _filter_fastest_at_max_context(cls, configs: list[dict]) -> list[dict]:
        """Keep only fastest passed config at max context per model."""
        # Group by model_id
        by_model = {}
        for cfg in configs:
            model = cfg["model_id"]
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(cfg)

        filtered = []
        for model, cfgs in by_model.items():
            # Find max context among passed configs
            passed = [c for c in cfgs if c.get("status") == "passed"]
            if not passed:
                continue
            max_ctx = max(c.get("context_length", 0) for c in passed)
            at_max = [c for c in passed if c.get("context_length", 0) == max_ctx]
            if at_max:
                # Keep fastest
                fastest = max(at_max, key=lambda c: cls._avg_gen_tok_s(c))
                filtered.append(fastest)
        return filtered

    @classmethod
    def _avg_gen_tok_s(cls, cfg: dict) -> float:
        """Extract average generation tok/s from metrics_json."""
        try:
            metrics = json.loads(cfg.get("metrics_json") or "[]")
            vals = [m.get("generation_tok_s") for m in metrics if m.get("success") and m.get("generation_tok_s")]
            return sum(vals) / len(vals) if vals else 0
        except Exception:
            return 0

    @classmethod
    def _build_system_prompt(cls, request: JudgeRequest) -> str:
        guidance = cls.USE_CASE_GUIDANCE.get(request.use_case, cls.USE_CASE_GUIDANCE["general"])
        return (
            "SYSTEM: You are an expert LLM optimization analyst. "
            f"Evaluate configurations for {request.use_case} on {cls.HARDWARE_DESCRIPTION}.\n"
            f"GUIDANCE: {guidance}\n"
            "Return ONLY valid JSON matching the verdict schema."
        )

    @classmethod
    def _build_summary_table(cls, configs: list[dict]) -> str:
        """Build markdown summary table for ALL configs."""
        headers = ["Model", "Config", "ctx", "tok/s", "prompt tok/s", "TTFT", "Quality", "VRAM", "Stability", "Key Tests"]
        rows = []

        for cfg in configs:
            model = cfg.get("model_id", "?")[:30]
            cfg_id = cfg.get("id", "?")[:8]
            ctx = cfg.get("context_length", "?")

            # Avg metrics
            gen_tok = cls._avg_gen_tok_s(cfg)
            prompt_tok = cls._avg_prompt_tok_s(cfg)
            ttft = cls._avg_ttft(cfg)

            # Quality
            qual = cls._extract_quality(cfg)

            # VRAM & Stability
            vram = cfg.get("peak_vram_gb", 0)
            stab = cfg.get("stability_score", 0)

            # Key tests summary
            key_tests = cls._format_key_tests(cfg)

            rows.append([
                model, cfg_id, str(ctx),
                f"{gen_tok:.1f}" if gen_tok else "—",
                f"{prompt_tok:.0f}" if prompt_tok else "—",
                f"{ttft:.0f}ms" if ttft else "—",
                f"{qual:.3f}" if qual else "—",
                f"{vram:.1f}GB" if vram else "—",
                f"{stab:.2f}" if stab else "—",
                key_tests,
            ])

        # Build markdown table
        table = "SUMMARY TABLE (ALL configs):\n"
        table += "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            table += "| " + " | ".join(row) + " |\n"
        return table

    @classmethod
    def _avg_prompt_tok_s(cls, cfg: dict) -> float:
        try:
            metrics = json.loads(cfg.get("metrics_json") or "[]")
            vals = [m.get("prompt_tok_s") for m in metrics if m.get("success") and m.get("prompt_tok_s")]
            return sum(vals) / len(vals) if vals else 0
        except Exception:
            return 0

    @classmethod
    def _avg_ttft(cls, cfg: dict) -> float:
        try:
            metrics = json.loads(cfg.get("metrics_json") or "[]")
            vals = [m.get("estimated_ttft_ms") for m in metrics if m.get("success") and m.get("estimated_ttft_ms")]
            return sum(vals) / len(vals) if vals else 0
        except Exception:
            return 0

    @classmethod
    def _extract_quality(cls, cfg: dict) -> float | None:
        try:
            qual = json.loads(cfg.get("quality_json") or "{}")
            return qual.get("overall")
        except Exception:
            return None

    @classmethod
    def _format_key_tests(cls, cfg: dict) -> str:
        try:
            qual = json.loads(cfg.get("quality_json") or "{}")
            by_test = qual.get("by_test", {})
            if by_test:
                parts = []
                for k, v in by_test.items():
                    if isinstance(v, (int, float)):
                        parts.append(f"{k}:{v:.2f}")
                return ", ".join(parts)[:50]
        except Exception:
            pass
        return "—"

    @classmethod
    def _build_top_n_detail(cls, configs: list[dict], top_n: int) -> str:
        """Build detailed section for top-N configs."""
        if not configs or top_n <= 0:
            return ""

        top_configs = configs[:top_n]
        parts = ["\nTOP " + str(top_n) + " CONFIGS — FULL DETAIL:\n"]

        for i, cfg in enumerate(top_configs, 1):
            model = cfg.get("model_id", "?")
            cfg_id = cfg.get("id", "?")[:8]
            ctx = cfg.get("context_length", "?")

            # Runtime config
            runtime = {}
            try:
                runtime = json.loads(cfg.get("config_json") or "{}")
            except Exception:
                pass

            gpu = runtime.get("gpu_ratio", "?")
            batch = runtime.get("eval_batch_size", "?")
            flash = runtime.get("flash_attention", "?")
            kv = "GPU" if runtime.get("offload_kv_cache_to_gpu") else "CPU"
            parallel = runtime.get("parallel", "?")
            checkpoints = runtime.get("context_checkpoints", "?")

            # Metrics
            gen_tok = cls._avg_gen_tok_s(cfg)
            prompt_tok = cls._avg_prompt_tok_s(cfg)
            ttft = cls._avg_ttft(cfg)

            # Quality
            qual = cls._extract_quality(cfg)
            by_test = {}
            try:
                by_test = json.loads(cfg.get("quality_json") or "{}").get("by_test", {})
            except Exception:
                pass

            stab = cfg.get("stability_score", 0)
            vram = cfg.get("peak_vram_gb", 0)

            # Sample prompt/output
            prompt_text = "—"
            output_text = "—"
            try:
                metrics = json.loads(cfg.get("metrics_json") or "[]")
                for m in metrics:
                    if m.get("success"):
                        prompt_text = (m.get("prompt_text") or "—")[:500]
                        output_text = (m.get("output_text") or "—")[:500]
                        break
            except Exception:
                pass

            parts.append(f"### Config {cfg_id} ({model})")
            parts.append(f"- Runtime: gpu={gpu}, batch={batch}, flash={flash}, kv={kv}, parallel={parallel}, checkpoints={checkpoints}")
            parts.append(f"- Metrics: gen_tok_s={gen_tok:.2f}, prompt_tok_s={prompt_tok:.0f}, ttft_ms={ttft:.0f}")
            parts.append(f"- Quality: overall={qual:.4f}" if qual else "- Quality: overall=N/A")
            if by_test:
                test_str = ", ".join(f"{k}:{v:.3f}" for k, v in by_test.items() if isinstance(v, (int, float)))
                parts.append(f"  {test_str}")
            parts.append(f"- Stability: {stab:.4f}, Peak VRAM: {vram:.2f} GB")
            parts.append(f"- Sample Prompt: {prompt_text}")
            parts.append(f"- Sample Output: {output_text}")
            parts.append("")

        return "\n".join(parts)

    @classmethod
    def _build_reasoning_matrix(cls, configs: list[dict]) -> str:
        """Build reasoning sweep matrix if configs have different reasoning settings."""
        # Group by base runtime (excluding reasoning)
        groups = {}
        for cfg in configs:
            try:
                rt = json.loads(cfg.get("config_json") or "{}")
                gen = rt.get("generation", {})
                reasoning = gen.get("reasoning", "off")
                # Create base key without reasoning
                base_key = json.dumps({k: v for k, v in rt.items() if k != "generation"})
                base_key += "|" + json.dumps({k: v for k, v in gen.items() if k != "reasoning"})
            except Exception:
                base_key = cfg.get("id", "")
                reasoning = "unknown"

            if base_key not in groups:
                groups[base_key] = {}
            groups[base_key][reasoning] = cfg

        # Only show groups with multiple reasoning levels
        multi_reasoning = {k: v for k, v in groups.items() if len(v) > 1}
        if not multi_reasoning:
            return ""

        parts = ["\nREASONING SETTINGS MATRIX (if available in run):\n"]
        parts.append("For each config tested with reasoning={off, low, medium, high, xhigh}:")
        parts.append("| Config | reasoning | tok/s | Quality | coding_task | structured_output | medium_reasoning |")
        parts.append("|--------|-----------|-------|---------|-------------|-------------------|------------------|")

        for base_key, variants in multi_reasoning.items():
            cfg_id = list(variants.values())[0].get("id", "?")[:8]
            for reasoning in ["off", "low", "medium", "high", "xhigh"]:
                if reasoning not in variants:
                    continue
                cfg = variants[reasoning]
                gen_tok = cls._avg_gen_tok_s(cfg)
                qual = cls._extract_quality(cfg)
                by_test = {}
                try:
                    by_test = json.loads(cfg.get("quality_json") or "{}").get("by_test", {})
                except Exception:
                    pass
                coding = by_test.get("coding_task", "—")
                structured = by_test.get("structured_output", "—")
                med_reas = by_test.get("medium_reasoning", "—")

                parts.append(
                    f"| {cfg_id} | {reasoning} | "
                    f"{gen_tok:.1f} | {qual:.3f} | {coding} | {structured} | {med_reas} |"
                )

        return "\n".join(parts) + "\n"

    @classmethod
    def _build_task_instruction(cls, request: JudgeRequest) -> str:
        """Build final task instruction with JSON schema."""
        schema = {
            "per_model_winner": {
                "model_id": "string",
                "config_id": "string",
                "reasoning": "string"
            },
            "overall_winner": {
                "model_id": "string",
                "config_id": "string",
                "reasoning": "string"
            },
            "task_breakdown": {
                "coding": {"winner": "string", "runner_up": "string", "reasoning": "string"},
                "chat": {"winner": "string", "runner_up": "string", "reasoning": "string"},
                "reasoning": {"winner": "string", "runner_up": "string", "reasoning": "string"},
                "structured": {"winner": "string", "runner_up": "string", "reasoning": "string"}
            },
            "avoid": [{"model_id": "string", "reason": "string"}],
            "warnings": ["string"],
            "confidence": 0.0
        }

        return (
            "\nTASK: Return JSON verdict with this exact schema:\n"
            f"{json.dumps(schema, indent=2)}\n"
            "Rules:\n"
            "- per_model_winner: best config for EACH model (if cross-model)\n"
            "- overall_winner: single best model+config across ALL models\n"
            "- task_breakdown: winners per capability area\n"
            "- avoid: models/configs that underperform on this hardware\n"
            "- warnings: hardware-specific cautions (BSOD risk, etc.)\n"
            "- confidence: 0.0-1.0 your confidence in this verdict\n"
        )