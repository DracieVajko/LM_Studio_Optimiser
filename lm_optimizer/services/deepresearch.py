"""DeepResearch Bench integration for LM Studio Optimizer.

This module provides integration with the DeepResearch Bench benchmark suite
for evaluating Deep Research Agents on 100 PhD-level research tasks.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DR_BENCH_DIR = Path("deep_research_bench")
DEFAULT_QUERY_FILE = DEFAULT_DR_BENCH_DIR / "data" / "prompt_data" / "query.jsonl"
DEFAULT_RAW_DATA_DIR = DEFAULT_DR_BENCH_DIR / "data" / "test_data" / "raw_data"
DEFAULT_CLEANED_DATA_DIR = DEFAULT_DR_BENCH_DIR / "data" / "test_data" / "cleaned_data"
DEFAULT_RESULTS_DIR = DEFAULT_DR_BENCH_DIR / "results"


@dataclass
class DeepResearchConfig:
    """Configuration for DeepResearch Bench evaluation."""
    model_name: str
    dr_bench_dir: Path = DEFAULT_DR_BENCH_DIR
    query_file: Path = DEFAULT_QUERY_FILE
    raw_data_dir: Path = DEFAULT_RAW_DATA_DIR
    cleaned_data_dir: Path = DEFAULT_CLEANED_DATA_DIR
    results_dir: Path = DEFAULT_RESULTS_DIR
    limit: int | None = None
    skip_cleaning: bool = False
    only_zh: bool = False
    only_en: bool = False
    force: bool = False
    max_workers: int = 5
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    jina_api_key: str | None = None
    llm_backend: str = "openrouter"  # "openrouter" or "openai"
    race_model: str = "openai/gpt-5.5"
    clean_model: str = "openai/gpt-5.6-luna"
    fact_model: str = "openai/gpt-5.4-mini"


async def ensure_dr_bench_repo(dr_bench_dir: Path) -> bool:
    """Ensure the DeepResearch Bench repository is cloned and up to date."""
    if not dr_bench_dir.exists():
        logger.info(f"Cloning DeepResearch Bench repository to {dr_bench_dir}")
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["git", "clone", "https://github.com/Ayanami0730/deep_research_bench.git", str(dr_bench_dir)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.error(f"Failed to clone repository: {result.stderr}")
                return False
            logger.info("Repository cloned successfully")
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return False
    else:
        # Pull latest changes
        logger.info("Updating existing DeepResearch Bench repository")
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["git", "pull"],
                cwd=str(dr_bench_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.warning(f"Git pull failed (non-critical): {result.stderr}")
        except Exception as e:
            logger.warning(f"Error updating repository (non-critical): {e}")
    return True


async def install_requirements(dr_bench_dir: Path) -> bool:
    """Install Python requirements for DeepResearch Bench."""
    req_file = dr_bench_dir / "requirements.txt"
    if not req_file.exists():
        logger.error(f"Requirements file not found at {req_file}")
        return False

    logger.info("Installing DeepResearch Bench requirements")
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.error(f"Failed to install requirements: {result.stderr}")
            return False
        logger.info("Requirements installed successfully")
    except Exception as e:
        logger.error(f"Error installing requirements: {e}")
        return False
    return True


async def run_model_on_queries(config: DeepResearchConfig, lm_studio_url: str) -> bool:
    """Run the local LM Studio model on all DeepResearch Bench queries.
    
    This generates the raw data files needed for evaluation.
    """
    logger.info(f"Running model {config.model_name} on DeepResearch Bench queries")

    # Load queries
    if not config.query_file.exists():
        logger.error(f"Query file not found: {config.query_file}")
        return False

    queries = []
    with open(config.query_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    queries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse query line: {e}")

    logger.info(f"Loaded {len(queries)} queries")

    # Apply limit
    if config.limit:
        queries = queries[:config.limit]
        logger.info(f"Limited to {len(queries)} queries")

    # Prepare output directory
    config.raw_data_dir.mkdir(parents=True, exist_ok=True)
    output_file = config.raw_data_dir / f"{config.model_name}.jsonl"

    # Check if output already exists
    if output_file.exists() and not config.force:
        logger.info(f"Output file already exists: {output_file}. Use --force to overwrite.")
        return True

    # Connect to LM Studio
    client = httpx.AsyncClient(base_url=lm_studio_url.rstrip("/"), timeout=300.0)

    try:
        # Load the model first
        load_config = {
            "model": config.model_name,
            "context_length": 32768,
            "flash_attention": True,
            "offload_kv_cache_to_gpu": True,
            "eval_batch_size": 512,
        }

        load_resp = await client.post("/api/v1/models/load", json=load_config)
        if load_resp.status_code != 200:
            logger.error(f"Failed to load model: {load_resp.text}")
            return False

        logger.info(f"Model {config.model_name} loaded successfully")

        # Process each query
        results = []
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}: {query['id']} ({query['language']})")

            prompt = query["prompt"]

            # Generate response using the model
            gen_config = {
                "model": config.model_name,
                "messages": [
                    {"role": "system", "content": "You are a deep research agent. Provide comprehensive, well-cited research reports."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 32768,
                "stream": False,
            }

            try:
                gen_resp = await client.post("/v1/chat/completions", json=gen_config, timeout=600.0)
                if gen_resp.status_code == 200:
                    data = gen_resp.json()
                    article = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    logger.error(f"Generation failed for query {query['id']}: {gen_resp.text}")
                    article = f"ERROR: Generation failed with status {gen_resp.status_code}"
            except Exception as e:
                logger.error(f"Generation error for query {query['id']}: {e}")
                article = f"ERROR: {e!s}"

            result = {
                "id": query["id"],
                "prompt": prompt,
                "article": article,
                "language": query["language"],
                "topic": query["topic"],
            }
            results.append(result)

            # Small delay to avoid overwhelming the server
            await asyncio.sleep(1.0)

        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(results)} results to {output_file}")
        return True

    finally:
        await client.aclose()
        # Unload model
        try:
            await client.post("/api/v1/models/unload", json={"model": config.model_name})
        except Exception:
            pass


async def run_race_evaluation(config: DeepResearchConfig) -> bool:
    """Run the RACE evaluation pipeline."""
    logger.info("Starting RACE evaluation")

    # Set environment variables for the evaluation
    env = os.environ.copy()
    if config.openrouter_api_key:
        env["OPENROUTER_API_KEY"] = config.openrouter_api_key
    if config.openai_api_key:
        env["OPENAI_API_KEY"] = config.openai_api_key
    if config.jina_api_key:
        env["JINA_API_KEY"] = config.jina_api_key
    env["LLM_BACKEND"] = config.llm_backend
    env["RACE_MODEL"] = config.race_model
    env["CLEAN_MODEL"] = config.clean_model
    env["FACT_MODEL"] = config.fact_model

    # Build command
    cmd = [
        sys.executable,
        str(config.dr_bench_dir / "deepresearch_bench_race.py"),
        config.model_name,
        "--raw_data_dir", str(config.raw_data_dir),
        "--cleaned_data_dir", str(config.cleaned_data_dir),
        "--max_workers", str(config.max_workers),
        "--query_file", str(config.query_file),
        "--output_dir", str(config.results_dir / "race" / config.model_name),
    ]

    if config.limit:
        cmd.extend(["--limit", str(config.limit)])
    if config.skip_cleaning:
        cmd.append("--skip_cleaning")
    if config.only_zh:
        cmd.append("--only_zh")
    if config.only_en:
        cmd.append("--only_en")
    if config.force:
        cmd.append("--force")

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            cwd=str(config.dr_bench_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hours max
        )

        if result.returncode != 0:
            logger.error(f"RACE evaluation failed: {result.stderr}")
            return False

        logger.info("RACE evaluation completed successfully")
        logger.info(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        return True

    except subprocess.TimeoutExpired:
        logger.error("RACE evaluation timed out")
        return False
    except Exception as e:
        logger.error(f"Error running RACE evaluation: {e}")
        return False


async def run_fact_evaluation(config: DeepResearchConfig) -> bool:
    """Run the FACT (citation) evaluation pipeline."""
    logger.info("Starting FACT evaluation")

    # Set environment variables
    env = os.environ.copy()
    if config.openrouter_api_key:
        env["OPENROUTER_API_KEY"] = config.openrouter_api_key
    if config.openai_api_key:
        env["OPENAI_API_KEY"] = config.openai_api_key
    if config.jina_api_key:
        env["JINA_API_KEY"] = config.jina_api_key
    env["LLM_BACKEND"] = config.llm_backend
    env["RACE_MODEL"] = config.race_model
    env["CLEAN_MODEL"] = config.clean_model
    env["FACT_MODEL"] = config.fact_model

    raw_data_path = config.raw_data_dir / f"{config.model_name}.jsonl"
    citation_output = config.results_dir / "fact" / config.model_name
    citation_output.mkdir(parents=True, exist_ok=True)

    # Pipeline steps
    steps = [
        ("extract", [
            sys.executable, "-m", "utils.extract",
            "--raw_data_path", str(raw_data_path),
            "--output_path", str(citation_output / "extracted.jsonl"),
            "--query_data_path", str(config.query_file),
            "--n_total_process", str(config.max_workers),
        ]),
        ("deduplicate", [
            sys.executable, "-m", "utils.deduplicate",
            "--raw_data_path", str(citation_output / "extracted.jsonl"),
            "--output_path", str(citation_output / "deduplicated.jsonl"),
            "--query_data_path", str(config.query_file),
            "--n_total_process", str(config.max_workers),
        ]),
        ("scrape", [
            sys.executable, "-m", "utils.scrape",
            "--raw_data_path", str(citation_output / "deduplicated.jsonl"),
            "--output_path", str(citation_output / "scraped.jsonl"),
            "--n_total_process", str(config.max_workers),
        ]),
        ("validate", [
            sys.executable, "-m", "utils.validate",
            "--raw_data_path", str(citation_output / "scraped.jsonl"),
            "--output_path", str(citation_output / "validated.jsonl"),
            "--query_data_path", str(config.query_file),
            "--n_total_process", str(config.max_workers),
        ]),
        ("stat", [
            sys.executable, "-m", "utils.stat",
            "--input_path", str(citation_output / "validated.jsonl"),
            "--output_path", str(citation_output / "fact_result.txt"),
        ]),
    ]

    for step_name, cmd in steps:
        logger.info(f"Running {step_name} step")
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                cwd=str(config.dr_bench_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=7200,
            )

            if result.returncode != 0:
                logger.error(f"{step_name} failed: {result.stderr}")
                return False

            logger.info(f"{step_name} completed")

        except subprocess.TimeoutExpired:
            logger.error(f"{step_name} timed out")
            return False
        except Exception as e:
            logger.error(f"Error running {step_name}: {e}")
            return False

    logger.info("FACT evaluation completed successfully")
    return True


async def run_deepresearch_benchmark(config: DeepResearchConfig, lm_studio_url: str) -> bool:
    """Run the complete DeepResearch Bench evaluation pipeline."""
    logger.info(f"Starting DeepResearch Bench evaluation for model: {config.model_name}")

    # Step 1: Ensure repository exists
    if not await ensure_dr_bench_repo(config.dr_bench_dir):
        return False

    # Step 2: Install requirements
    if not await install_requirements(config.dr_bench_dir):
        return False

    # Step 3: Run model on queries (generate raw data)
    if not await run_model_on_queries(config, lm_studio_url):
        return False

    # Step 4: Run RACE evaluation
    if not await run_race_evaluation(config):
        return False

    # Step 5: Run FACT evaluation
    if not await run_fact_evaluation(config):
        return False

    logger.info("DeepResearch Bench evaluation completed successfully!")
    return True


def print_results_summary(config: DeepResearchConfig) -> None:
    """Print a summary of the evaluation results."""
    race_result_file = config.results_dir / "race" / config.model_name / "race_result.txt"
    fact_result_file = config.results_dir / "fact" / config.model_name / "fact_result.txt"

    if race_result_file.exists():
        print("\n=== RACE Evaluation Results ===")
        with open(race_result_file) as f:
            print(f.read())

    if fact_result_file.exists():
        print("\n=== FACT Evaluation Results ===")
        with open(fact_result_file) as f:
            print(f.read())
