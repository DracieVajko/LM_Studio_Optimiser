# Troubleshooting Guide

## Common Issues

### 1. LM Studio Connection Failed

**Symptoms:**
- `lm-optimizer status` shows "LM Studio: OFFLINE"
- Connection timeout errors
- `httpx.ConnectError`

**Solutions:**
1. Verify LM Studio is running
2. Enable "Developer API" in LM Studio Settings → Developer
3. Check the port (default 1234) matches your `.env` (`LM_STUDIO_URL`)
4. Ensure no firewall blocking the connection
5. Try `curl http://127.0.0.1:1234/api/v1/models` to test locally, or `curl http://192.168.1.100:1234/api/v1/models` for remote

**Remote Access (Tailscale/VPN):**
- Ensure VPN is running on both machines if using remote URL
- Use the remote IP (e.g., `http://192.168.1.100:1234` or `http://100.x.x.x:1234` for Tailscale) not localhost
- For remote access, LM Studio must bind to `0.0.0.0:1234` — see Security notes in README
- Validate URL via Settings → Test Connection in Web UI or `lm-optimizer status --url <url>`

### 2. Model Not Found

**Symptoms:**
- "Model not found" error
- Model appears in LM Studio but not in optimizer

**Solutions:**
1. Run `lm-optimizer models` to see exact model IDs
2. Model IDs are case-sensitive - use exact ID from list
3. Refresh LM Studio model list (restart LM Studio if needed)
4. Some models need to be "loaded" once in LM Studio UI before appearing in API

### 3. Out of Memory (OOM) Errors

**Symptoms:**
- `CUDA out of memory` errors
- Benchmark fails with OOM
- System becomes unresponsive

**Solutions:**
1. Reduce GPU ratio: `--gpu-ratio 0.6` or lower
2. Move KV cache to CPU: `--kv-cpu`
3. Reduce context length: `--context 4096`
4. Reduce batch size: `--batch 128`
5. Disable Flash Attention: `--no-flash`
6. Set `HW_GPU_VRAM_GB` slightly lower than actual (e.g., 5.5 for 6GB) to leave headroom

### 4. Quality Scores Too Low

**Symptoms:**
- "Quality X below threshold Y" messages
- Configurations rejected despite good speed

**Solutions:**
1. Lower `MINIMUM_QUALITY_SCORE` in `.env` (e.g., 0.95 for speed profile)
2. Some models have non-deterministic outputs - quality scoring may be unreliable
3. Check if model supports `seed` parameter for reproducibility
4. Use `--profile speed` which has lower quality threshold (0.95)

### 5. Optimization Takes Too Long

**Symptoms:**
- Running for hours
- Seems stuck on one configuration

**Solutions:**
1. Reduce `BENCHMARK_RUNS` (default 3) to 2
2. Reduce `BENCHMARK_TIMEOUT` 
3. Use `--profile speed` which tests fewer configurations
4. Check logs for repeated failures causing retries

### 6. Web UI Not Loading

**Symptoms:**
- Browser shows "Cannot connect"
- Port 8080 already in use

**Solutions:**
1. Change `WEB_UI_PORT` in `.env`
2. Ensure `WEB_UI_ENABLED=true`
3. Check firewall allows localhost:8080
4. Run `python -m lm_optimizer.web_main` directly to see errors

### 7. Preset Not Applied

**Symptoms:**
- "No preset found" error
- Model loads but with wrong config

**Solutions:**
1. Run `lm-optimizer optimize` first to create preset
2. Check `profiles/` directory for preset files
3. Model ID must match exactly (including org prefix like `user/model`)
4. Use `lm-optimizer apply --dry-run` to verify config

### 8. Flash Attention Errors

**Symptoms:**
- "Flash Attention not supported" 
- Model fails to load with flash_attention=true

**Solutions:**
1. Disable Flash Attention: `--no-flash`
2. Some models/quantizations don't support Flash Attention
3. LM Studio version may not expose this parameter
4. Check `lm-optimizer inspect` output for supported parameters

### 9. MoE Expert Count Issues

**Symptoms:**
- "Invalid num_experts" error
- Model fails to load with custom expert count

**Solutions:**
1. Only use expert counts supported by the model
2. Common values: 8 (Mixtral), 64 (DeepSeek-MoE)
3. Check `lm-optimizer inspect` for detected expert count
4. Leave `num_experts` unset for default

### 10. Benchmark Results Inconsistent

**Symptoms:**
- High variance between runs
- Speed fluctuates significantly

**Solutions:**
1. Increase `BENCHMARK_RUNS` to 5
2. Close other GPU applications
3. Ensure laptop is plugged in (not on battery)
4. Disable Windows "Game Mode" and GPU scheduling
5. Set GPU to "Maximum Performance" in NVIDIA Control Panel

## Debugging

### Enable Verbose Logging

```bash
lm-optimizer --verbose status
```

Or set in `.env`:
```
LOG_LEVEL=DEBUG
LOG_FORMAT=text
```

### Check Logs

```bash
# View structured logs
cat data/logs/optimizer.log | jq .

# View recent results
ls -la data/reports/
```

### Manual API Testing

```bash
# List models (local)
curl http://127.0.0.1:1234/api/v1/models

# Or remote
curl http://192.168.1.100:1234/api/v1/models

# Load model
curl -X POST http://127.0.0.1:1234/api/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "model-name", "config": {"context_length": 4096}}'

# Generate
curl -X POST http://127.0.0.1:1234/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "model-name", "messages": [{"role": "user", "content": "Hello"}]}'

# Unload
curl -X POST http://127.0.0.1:1234/api/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"identifier": "abc123"}'
```

### GPU Monitoring

```bash
# Watch GPU usage
nvidia-smi -l 1

# Or use GPUtil in Python
python -c "import GPUtil; [print(g) for g in GPUtil.getGPUs()]"
```

## Performance Tips

### Example: Lower VRAM System (e.g., 6GB Laptop GPU)

| Model Size | Recommended GPU Ratio | KV Cache | Context |
|------------|----------------------|----------|---------|
| 7B Q4      | 0.9-1.0              | GPU      | 8192    |
| 7B Q5      | 0.7-0.8              | GPU      | 8192    |
| 7B Q6      | 0.6-0.7              | CPU      | 8192    |
| 13B Q4     | 0.5-0.6              | CPU      | 4096    |
| 13B Q5     | 0.4-0.5              | CPU      | 4096    |

### System Optimization

1. **Windows**: Disable "Hardware-accelerated GPU scheduling"
2. **Power**: High Performance power plan, plugged in
3. **Background**: Close browsers, games, other GPU apps
4. **Drivers**: Latest NVIDIA Studio drivers

## Getting Help

1. Check `data/logs/optimizer.log` for detailed errors
2. Run with `--verbose` flag
3. Review `data/reports/` JSON files for failed config details
4. Open GitHub issue with:
   - LM Studio version
   - Model ID
   - Hardware specs
   - Error logs
   - `.env` config (redact URLs)