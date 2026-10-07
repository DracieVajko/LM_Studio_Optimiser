# LM Studio Auto Optimizer - Next Steps Roadmap

This document outlines the prioritized roadmap for future development. Designed for frontier AI systems to continue development seamlessly.

---

## 🎯 Immediate (v1.5.3) - Ollama Live Verification

### Goal: Production-ready Ollama backend

**Blockers to resolve:**
- [ ] Live Ollama server verification (currently mocked-test only)
- [ ] `ollama-export` / `ollama-apply` E2E test with real Ollama
- [ ] Ollama parameter sweep validation (num_ctx, num_batch, num_gpu, num_thread)
- [ ] Ollama sampling sweep (top_p, top_k, min_p, temperature)
- [ ] Ollama Modelfile versioning and diff tracking

**Files to modify:**
- `lm_optimizer/backends/ollama/client.py` - fix any live issues
- `lm_optimizer/backends/ollama/modelfile.py` - validate roundtrip
- `docs/OLLAMA_LIVE_CHECK.md` - update with verified commands

**Test commands:**
```bash
ollama serve
lm-optimizer --backend ollama status
lm-optimizer --backend ollama models
lm-optimizer --backend ollama optimize <model> --profile balanced
lm-optimizer ollama-export --model <base> --param num_ctx=8192 -o test.modelfile.md
lm-optimizer ollama-apply --file test.modelfile.md --as <tag>:opt --yes
```

---

## 🎯 Short-term (v1.6.0) - Model Judge AI Enhancement

### Goal: Real AI-assisted model selection

**Current state:** Heuristic judge only (rule-based)
**Target:** Local model judge + API judge backends

**Implementation:**
1. **Local Judge Backend** (`lm_optimizer/services/model_judge.py`)
   - Use LM Studio/Ollama to run a reasoning model as judge
   - Prompt template with run data, hardware, use case
   - Parse structured JSON response

2. **API Judge Backend** 
   - OpenRouter / OpenAI-compatible endpoint
   - Configurable model (e.g., `openrouter/auto`, `gpt-4o-mini`)
   - Rate limiting and cost tracking

3. **Judge Prompt Engineering**
   - Include: all configs with metrics, hardware constraints, use case, priority
   - Output: structured verdict with confidence, reasoning, alternatives
   - Few-shot examples from historical data

**Judge Prompt Template:**
```
You are an expert LLM inference engineer. Given optimization results, recommend the best configuration.

Hardware: {vram_gb}GB VRAM, {ram_gb}GB RAM, GPU: {gpu_name}
Use case: {use_case} | Priority: {priority}
Model: {model_id} ({architecture})

Configurations tested:
{config_table}

Constraints: VRAM ≤ {vram_limit}GB, Quality ≥ 0.95

Return JSON:
{
  "recommended_config_id": "...",
  "reasoning": "...",
  "confidence": 0.9,
  "pros": [...],
  "cons": [...],
  "alternatives": ["...", "..."],
  "warnings": [...]
}
```

---

## 🎯 Medium-term (v1.7.0) - Campaign & Comparison Automation

### Goal: Automated multi-model campaigns with verdicts

**Features:**
- [ ] `campaign` command: run optimization across model list with resume
- [ ] Automatic leaderboard generation (Markdown + HTML)
- [ ] Pairwise comparison matrix (A/B duels for top N configs)
- [ ] Cherry-pick workflow: run campaign → judge selects best per model → final comparison
- [ ] Campaign state persistence (pause entire campaign, resume later)

**CLI:**
```bash
lm-optimizer campaign --models models.txt --profile balanced --overnight
lm-optimizer campaign --resume  # continue from last incomplete
lm-optimizer compare-leaderboard --output results/campaign/leaderboard.md
```

---

## 🎯 Medium-term (v1.7.0) - Advanced Search Space

### Goal: Deeper, smarter parameter exploration

**Current limits:** ~25 probes max, bounded grid
**Target:** Adaptive exploration with learned priors

**Enhancements:**
- [ ] Bayesian optimization for continuous params (temperature, top_p, gpu_ratio)
- [ ] Learned search space from historical runs (per-architecture priors)
- [ ] Multi-objective optimization (Pareto front tracking across runs)
- [ ] Dynamic budget allocation (more probes for promising regions)
- [ ] Transfer learning: use Model A's results to seed Model B's search

---

## 🎯 Long-term (v2.0.0) - Distributed & Cloud

### Goal: Scale beyond single machine

**Architecture:**
- [ ] Job queue (Redis/RabbitMQ) for distributed workers
- [ ] Worker registration with hardware capabilities
- [ ] Central coordinator for search space partitioning
- [ ] Result aggregation and leaderboard
- [ ] Cloud GPU integration (RunPod, Lambda, Vast.ai)

**API:**
```python
# Submit job to cluster
client.submit_optimization(model, profile, hardware_requirements)
# Poll for results
client.get_job_status(job_id)
# Get aggregated results
client.get_cluster_leaderboard()
```

---

## 🎯 Long-term (v2.0.0) - Continuous Optimization

### Goal: Always-optimal configurations

**Concept:** Monitor model updates, hardware changes, usage patterns → re-optimize automatically

**Components:**
- [ ] Model version detection (hash-based change detection)
- [ ] Hardware change detection (GPU swap, driver update)
- [ ] Usage analytics (which configs actually used in production)
- [ ] Scheduled re-optimization (cron-like, with change triggers)
- [ ] A/B rollout for new configs (gradual traffic shift)

---

## 🔬 Research / Experimental

### Speculative Decoding Optimization
- [ ] Draft model selection optimization (not just discovery)
- [ ] Speculative token count sweep per draft
- [ ] MTP (Multi-Token Prediction) layer optimization

### MoE (Mixture of Experts) Deep Optimization
- [ ] Expert routing analysis (which experts activate for which tasks)
- [ ] Expert pruning / merging based on activation patterns
- [ ] Dynamic expert count per layer

### Quantization-Aware Optimization
- [ ] Joint optimization of quantization + inference params
- [ ] Per-layer quantization sensitivity analysis
- [ ] AWQ / GPTQ calibration data generation from benchmark suite

### Reasoning Model Optimization
- [ ] `reasoning_budget_message` optimization
- [ ] Chain-of-thought length vs quality tradeoff
- [ ] Reasoning trace compression

---

## 📋 Technical Debt & Maintenance

### High Priority
- [ ] Migrate from `orjson` to stdlib `json` where performance not critical (reduce deps)
- [ ] Type stub generation for all public APIs
- [ ] OpenAPI spec generation from FastAPI routes
- [ ] Structured logging migration (structlog → standard logging + JSON formatter)

### Medium Priority
- [ ] Database migration system (currently schema is implicit)
- [ ] Configuration versioning (track schema changes)
- [ ] Plugin system for custom benchmark cases / quality checks
- [ ] Web UI: real-time collaboration (multiple users viewing same run)

### Low Priority
- [ ] macOS native packaging (py2app / briefcase)
- [ ] Windows installer (MSIX / winget)
- [ ] Homebrew formula
- [ ] VS Code extension for results viewing

---

## 🧪 Testing Infrastructure

### Add to CI
- [ ] Ollama live test (start Ollama in CI, run basic optimize)
- [ ] Multi-GPU test (if runners available)
- [ ] Web UI E2E with Playwright (full optimize flow)
- [ ] Performance regression benchmarks (track tok/s over time)

### Property-Based Testing
- [ ] Hypothesis tests for search space generation
- [ ] Fuzzing for parameter validation
- [ ] Invariant testing for optimizer state machine

---

## 📚 Documentation

### Needed
- [ ] `docs/DEVELOPER_GUIDE.md` - how to add backends, phases, tests
- [ ] `docs/API_REFERENCE.md` - generated from OpenAPI spec
- [ ] `docs/JUDGE_PROMPTS.md` - prompt engineering guide
- [ ] Video tutorials for key workflows

---

## 🎯 Model-Specific Optimization Targets (User Requested)

### Immediate Testing Queue
1. **bonsai27b** vs **ministral3b** - small model comparison
2. **20B+ models** - cherry-pick best configs
3. **MoE models** (mixtral, etc.) - expert gating validation
4. **Deep research models** - long-context optimization

### Optimization Campaign Plan
```bash
# Phase 1: Small models (< 8B)
lm-optimizer auto --skip large --profile balanced

# Phase 2: Medium models (8B-20B)  
lm-optimizer auto --only medium --profile balanced

# Phase 3: Large models (20B+)
lm-optimizer auto --only large --profile quality

# Phase 4: Specialized
lm-optimizer context-sweep --model <best_20b> --step 1000
lm-optimizer sample-sweep --model <best_overall> --config best.json
```

---

## 📝 Notes for Next AI

### Key Design Principles
1. **User decides, AI assists** - never automate final decisions
2. **Checkpoint everything** - resume must be zero-rework
3. **Hardware-agnostic** - no hardcoded GPU assumptions
4. **Empirical over theoretical** - measure, don't predict
5. **Graceful degradation** - Web UI works without LM Studio

### Common Pitfalls to Avoid
- Don't add hardcoded model names (use capabilities detection)
- Don't skip quality validation (failed configs never win)
- Don't break resume (checkpoint format must be forward-compatible)
- Don't mix sandbox duels with optimization runs (separate tables)

### Debugging Tips
- Check `data/logs/` for structured JSON logs
- Use `lm-optimizer checkpoints` to see resume state
- Web UI `/api/runs/{id}/configurations` for detailed config data
- `lm-optimizer param-matrix` for capability verification

---

*Generated: 2026-10-07 | Version: 1.5.2*