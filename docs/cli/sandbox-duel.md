# sandbox-duel

Compare two models on the same prompt, each using its own validated winner
configuration.

```
lm-optimizer sandbox-duel --model-a ID --model-b ID --prompt "..."
```

## How it measures

For each side: load with that model's winner config, run the prompt N times,
unload. `BenchmarkService.run_cases` owns the load/verify/unload skeleton, so
the host never holds two models at once and a failed repetition cannot poison
the next one.

Medians (not means) are reported — a single slow repetition should not move the
verdict.

## Config source

`auto-best` is the only supported source today: the validated winner
(`best_config_id` → a `passed` configuration) from the model's most recent
successful run.

The command **refuses** to duel when a model has no validated winner, rather
than guessing a configuration. Optimize the model first.

## Metrics and verdict

| Metric | Direction |
|--------|-----------|
| TTFT | lower wins |
| Generation speed | higher wins |
| Quality | higher wins |

A metric with no measurement yields no winner — an absent measurement is never
reported as a win. If neither side has a usable speed measurement, the report
says the overall winner is undetermined.

## Options

| Option | Default | Meaning |
|--------|---------|---------|
| `--model-a` / `--model-b` | required | model ids |
| `--prompt` | required | the prompt both models answer |
| `--repetitions` | 3 | runs per model |
| `--style` | `balanced` | benchmark style |
| `--quality` | 0.97 | quality threshold |
| `--output` | `results/duels` | report directory |

## Output

`results/duels/duel-<a>-vs-<b>-YYYYMMDD-HHMMSS.md` with the configuration table,
the prompt, per-run results, medians, a per-metric verdict, and the environment.

## Example

```bash
lm-optimizer sandbox-duel \
  --model-a qwen/qwen3-coder-30b \
  --model-b mistralai/ministral-3-14b-reasoning \
  --prompt "Write a Python function that returns the n-th Fibonacci number" \
  --repetitions 5
```