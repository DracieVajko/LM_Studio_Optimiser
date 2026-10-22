// LM Studio Auto Optimizer - Results Page

import { API } from './api.js?v=8';
import { UI } from './ui.js?v=8';
import { Charts } from './charts.js?v=8';
import { WebSocketManager } from './websocket.js?v=8';

const ResultsPage = {
    state: {
        runId: null,
        run: null,
        configurations: [],
        paretoConfigs: [],
        currentChart: 'generation-context',
    },

    async init() {
        // Get run ID from URL
        this.state.runId = window.location.pathname.split('/').pop();
        if (!this.state.runId) {
            window.location.href = '/history';
            return;
        }

        try {
            UI.init();
        } catch (e) {
            console.error('Results init failed:', e);
        }
        const ok = await this.loadRun();
        if (!ok) return; // error card already rendered; never a spinner
        try {
            this.render();
            this.loadGenerationProfiles();
        } catch (e) {
            console.error('Results render failed:', e);
            document.getElementById('main-content').innerHTML = UI.renderRunError(
                this.state.runId, 'client', (e && e.message) || String(e)
            );
            return;
        }
        this.attachEvents();
        this.connectLive();
    },

    connectLive() {
        // Actually connect the Web UI to the backend WebSocket for live runs.
        const status = (this.state.run?.status || '').toLowerCase();
        if (!['running', 'resumed', 'paused'].includes(status)) return;
        try {
            this.state.ws = new WebSocketManager();
            this.state.ws.connect(this.state.runId, (msg) => this.onLiveMessage(msg));
        } catch (e) {
            console.warn('Live connection failed', e);
        }
    },

    onLiveMessage(msg) {
        if (!msg || !msg.type) return;
        if (msg.type === 'progress') {
            this.state.live = msg;
            const el = document.getElementById('live-progress');
            if (el) el.innerHTML = this.renderLiveProgressInner(msg);
            const log = document.getElementById('live-log-events');
            if (log && msg.events) this.appendLiveLog(msg.events);
        } else if (msg.type === 'log') {
            this.appendLiveLog(msg.events || []);
        } else if (msg.type === 'complete') {
            UI.showToast('Optimization complete — reloading', 'success');
            setTimeout(() => window.location.reload(), 1500);
        } else if (msg.type === 'error') {
            UI.showToast(msg.error || 'Optimization error', 'error');
        }
    },

    appendLiveLog(events) {
        const log = document.getElementById('live-log-events');
        if (!log || !events) return;
        for (const ev of events) {
            const div = document.createElement('div');
            div.className = 'font-mono text-xs py-0.5';
            const ts = (ev.ts || '').slice(11, 19) || '';
            div.textContent = `[${ts}] ${ev.message || ''}${ev.detail ? ' — ' + ev.detail : ''}`;
            log.appendChild(div);
            while (log.children.length > 100) log.removeChild(log.firstChild);
        }
        log.scrollTop = log.scrollHeight;
    },

    renderLiveSection() {
        const status = (this.state.run?.status || '').toLowerCase();
        const isLive = ['running', 'resumed', 'paused'].includes(status);
        if (!isLive) return '';
        const live = this.state.live || {};
        return `
            <div class="card" id="live-progress-card">
                <div class="card-header"><h3 class="font-semibold">Live Optimization</h3></div>
                <div class="card-body">
                    <div id="live-progress">${this.renderLiveProgressInner(live)}</div>
                    <div class="mt-4">
                        <h4 class="font-medium text-sm mb-2">Live log</h4>
                        <div id="live-log-events" class="bg-gray-900 text-green-300 rounded p-3 h-40 overflow-y-auto"></div>
                        <details class="mt-2 text-xs text-gray-500">
                            <summary class="cursor-pointer">Detailed logs</summary>
                            <div id="live-log-detail" class="font-mono whitespace-pre-wrap mt-1">Verbose backend logs are in data/logs. UI shows compact events only.</div>
                        </details>
                    </div>
                </div>
            </div>
        `;
    },

    renderPhaseSteps(stage) {
        const steps = [
            ['speed_discovery', '1. Speed'],
            ['quality_check', '2. Quality'],
            ['recovery', '3. Recovery'],
            ['final_validation', '4. Validation'],
            ['context_optional', '5. Context (opt)'],
        ];
        const active = steps.findIndex(([key]) => key === stage);
        return `<div class="flex flex-wrap gap-2 mb-3 text-xs">` + steps.map(([key, label], i) => {
            const cls = i === active
                ? 'bg-blue-600 text-white'
                : (active > 0 && i < active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-500');
            return `<span class="px-2 py-1 rounded font-mono ${cls}">${label}</span>`;
        }).join('') + `</div>`;
    },

    renderLiveProgressInner(p) {
        const fmt = (v, suffix='') => (v === null || v === undefined || v === '' ? '—' : `${v}${suffix}`);
        const cur = p.current_config ? `ctx ${p.current_config.context_length ?? '—'}, flash ${p.current_config.flash_attention ? 'ON' : 'OFF'}, KV ${p.current_config.offload_kv_cache_to_gpu ? 'GPU' : 'CPU'}, batch ${p.current_config.eval_batch_size ?? '—'}` : '—';
        return `
            ${this.renderPhaseSteps(p.stage)}
            <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div><div class="text-gray-500">Stage</div><div class="font-mono font-medium">${fmt(p.stage)}</div></div>
                <div><div class="text-gray-500">Model</div><div class="font-mono font-medium break-all">${fmt(p.model || this.state.run?.model?.id)}</div></div>
                <div class="col-span-2"><div class="text-gray-500">Current configuration</div><div class="font-mono font-medium">${cur}</div></div>
                <div><div class="text-gray-500">Candidates tested</div><div class="font-mono font-medium">${fmt(p.configs_tested)} / ${fmt(p.configs_total)}</div></div>
                <div><div class="text-gray-500">Successful / Failed / OOM</div><div class="font-mono font-medium">${fmt(p.configs_passed)} / ${fmt(p.configs_failed)} / ${fmt(p.configs_oom)}</div></div>
                <div><div class="text-gray-500">Current best</div><div class="font-mono font-medium">${fmt(p.best_score)}</div></div>
                <div><div class="text-gray-500">Current speed</div><div class="font-mono font-medium">${fmt(p.current_speed, ' tok/s')}</div></div>
                <div><div class="text-gray-500">VRAM / RAM</div><div class="font-mono font-medium">${fmt(p.vram_gb, ' GB')} / ${fmt(p.ram_gb, ' GB')}</div></div>
                <div><div class="text-gray-500">Elapsed</div><div class="font-mono font-medium">${fmt(p.elapsed_s, 's')}</div></div>
                <div class="col-span-2"><div class="text-gray-500">Context boundary progress</div><div class="font-mono font-medium">${(p.context_progress || []).join(', ') || '—'}</div></div>
                <div class="col-span-2"><div class="text-gray-500">Estimated remaining</div><div class="font-mono font-medium">${fmt(p.remaining || 'adaptive / unknown')}</div></div>
            </div>
        `;
    },

    renderBestConfigs() {
        // Kept for backward compat; header now owns the recommendation block.
        return '';
    },

    async loadRun() {
        const main = document.getElementById('main-content');
        try {
            this.state.run = await API.getRun(this.state.runId);
            this.state.configurations = await API.getConfigurations(this.state.runId);
            this.state.paretoConfigs = await API.getParetoFrontier(this.state.runId);

            // Convert to chart-friendly format
            this.state.configurations = this.state.configurations.configurations || this.state.configurations;
            this.state.paretoConfigs = this.state.paretoConfigs.configurations || this.state.paretoConfigs;
            if (!this.state.run) throw new Error('Empty run response');
            return true;
        } catch (error) {
            console.error('Failed to load run:', error);
            const status = (error && error.status) || 'network/timeout';
            const reason = (error && error.message) || String(error);
            if (main) main.innerHTML = UI.renderRunError(this.state.runId, status, reason);
            else UI.showToast('Failed to load results', 'error');
            return false;
        }
    },

    render() {
        const main = document.getElementById('main-content');
        main.innerHTML = `
            <div class="space-y-8">
                ${this.renderLiveSection()}
                ${this.renderHeader()}
                ${this.renderTabs()}
                <div id="tab-content" class="card">
                    <div class="card-body" id="tab-panel">
                        ${this.renderChartTab()}
                    </div>
                </div>
                ${this.renderConfigTable()}
            </div>
        `;

        // Render initial chart
        setTimeout(() => this.renderChart(), 100);
    },

    renderHeader() {
        const run = this.state.run;
        const passed = this.state.configurations.filter(c => c.status === 'passed');
        const failed = this.state.configurations.filter(c => c.status !== 'passed');
        const bestConfig = run.best_config_id
            ? this.state.configurations.find(c => c.id === run.best_config_id)
            : passed.sort((a, b) => (b.score || 0) - (a.score || 0))[0] || null;
        const oom = this.state.configurations.filter(c => c.status === 'oom').length;
        const timeouts = this.state.configurations.filter(c => c.status === 'timeout').length;
        const qualityRejected = this.state.configurations.filter(c => c.status === 'quality_failed').length;
        const loadFailures = failed.length - oom - timeouts - qualityRejected;

        // ZERO-PASS: explicit failure state, no Apply offer.
        if (!bestConfig) {
            return `
            <div class="card border-red-300">
                <div class="card-header"><h1 class="text-2xl font-bold text-red-700">Optimization unsuccessful</h1>
                <p class="text-sm text-gray-500 font-mono">${run.model?.id || ''}</p></div>
                <div class="card-body">
                    <p>No valid configurations were successfully benchmarked.</p>
                    <p class="mt-2 text-sm">Reason: load_failures=${loadFailures}, oom=${oom}, timeouts=${timeouts}, quality_rejected=${qualityRejected}</p>
                    <div class="flex flex-wrap gap-2 mt-2">
                        <span class="badge badge-info">${run.profile}</span>
                        <span class="badge ${UI.getStatusBadge(run.status)}">${run.status}</span>
                    </div>
                    <p class="mt-2 text-xs text-gray-500">Apply is disabled: there is no validated configuration.</p>
                </div>
            </div>`;
        }

        const isPartial = failed.length > 0;
        return `
            <div class="card">
                <div class="card-header flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                    <div>
                        <div class="flex items-center gap-3">
                            <a href="/history" class="text-gray-400 hover:text-gray-600">
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path>
                                </svg>
                            </a>
                            <div>
                                <h1 class="text-2xl font-bold text-gray-900">${isPartial ? 'Optimization partially completed' : 'FINAL RECOMMENDATION'}</h1>
                                <p class="text-sm text-gray-500 font-mono">${run.model?.id || ''} — ${run.profile} — ${run.status}</p>
                                ${isPartial ? `<p class="text-sm">Successful: ${passed.length} &nbsp; Failed: ${failed.length}</p>` : ''}
                            </div>
                        </div>
                        <div class="flex flex-wrap gap-2 mt-2">
                            <span class="badge badge-info">${run.profile}</span>
                            <span class="badge ${UI.getStatusBadge(run.status)}">${run.status}</span>
                            <span class="badge badge-gray">${run.duration_seconds ? run.duration_seconds.toFixed(1) + 's' : '—'}</span>
                        </div>
                    </div>
                    <div class="flex gap-2">
                        ${this.renderPauseResumeButtons(run.status)}
                        <button id="apply-best-btn" class="btn btn-success">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                            </svg>
                            Apply Best Config
                        </button>
                        <button id="export-btn" class="btn btn-outline">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4-4m4 4V4"></path>
                            </svg>
                            Export
                        </button>
                    </div>
                </div>

                <div class="card-body mt-4">
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                        ${this.renderMetricCard('Generation Speed', (bestConfig?.avg_generation_tok_s?.toFixed(1) + ' tok/s') || '—', '🚀')}
                        ${this.renderMetricCard('Prompt Speed', (bestConfig?.avg_prompt_tok_s?.toFixed(0) + ' tok/s') || '—', '⚡')}
                        ${this.renderMetricCard('TTFT', (bestConfig?.avg_ttft_ms?.toFixed(0) + ' ms') || '—', '⏱️')}
                        ${this.renderMetricCard('Quality', bestConfig?.quality?.overall?.toFixed(3) || '—', '✨')}
                    </div>

                    ${this.renderBestConfigDetails(bestConfig)}
                    ${this.renderPhaseABStrip(bestConfig)}
                    ${this.renderAlternatives(bestConfig)}
                    <div class="mt-4 p-3 bg-gray-50 rounded" id="gen-profiles">
                        <h4 class="font-medium mb-1">GENERATION PROFILES</h4>
                        <p class="text-xs text-gray-500">Loading recommendations...</p>
                    </div>
                </div>
            </div>
        `;
    },

    renderPhaseABStrip(bestConfig) {
        const run = this.state.run || {};
        const cfgs = this.state.configurations || [];
        const raw = run.raw_fastest_config_id
            ? cfgs.find(c => c.id === run.raw_fastest_config_id)
            : null;
        const pb = run.phase_b_result || null;
        const paused = run.pause || null;
        if (!raw && !pb && !paused) return '';
        const tok = (c) => c && c.avg_generation_tok_s != null ? c.avg_generation_tok_s.toFixed(1) + ' tok/s' : '—';
        return `<div class="mt-4 p-3 bg-amber-50 rounded border border-amber-200">
            <h4 class="font-medium mb-1">PHASE A/B TRANSPARENCY</h4>
            ${paused ? `<div class="text-sm"><span class="font-semibold text-amber-700">PAUSED</span> <span class="font-mono">${paused.phase ?? '?'} / ${paused.reason ?? '?'}${paused.paused_at ? ' @ ' + paused.paused_at : ''} — resume possible: YES (same run ID)</span></div>` : ''}
            ${raw ? `<div class="text-sm"><span class="text-gray-500">Raw fastest${raw.id === bestConfig?.id ? ' (= winner)' : ' (rejected for quality)'}:</span> <span class="font-mono">${raw.context_length} ctx, ${tok(raw)}</span></div>` : ''}
            <div class="text-sm"><span class="text-gray-500">Quality-safe winner:</span> <span class="font-mono">${bestConfig?.context_length} ctx, ${tok(bestConfig)}</span></div>
            ${pb ? `<div class="text-sm mt-1"><span class="text-gray-500">Phase B:</span> <span class="font-mono">GPU-only ${pb.gpu_only_max ?? '—'} / system ${pb.system_max ?? '—'} / final ${pb.final_capacity ?? '—'}</span></div>` : ''}
        </div>`;
    },

    renderPauseResumeButtons(status) {
        const isRunning = ['running', 'resumed'].includes(status);
        const isPaused = status === 'paused';
        if (!isRunning && !isPaused) return '';
        
        if (isRunning) {
            return `
                <button id="pause-run-btn" class="btn btn-warning" title="Pause at next safe boundary">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    Pause
                </button>`;
        }
        if (isPaused) {
            return `
                <button id="resume-run-btn" class="btn btn-success" title="Resume from checkpoint">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path>
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    Resume
                </button>`;
        }
        return '';
    },

    renderAlternatives(best) {
        const passed = this.state.configurations.filter(c => c.status === 'passed');
        if (!passed.length) return '';
        const by = (fn) => passed.slice().sort((a, b) => fn(b) - fn(a))[0];
        const speed = by(c => c.avg_generation_tok_s || 0);
        const ctx = by(c => c.context_length || 0);
        const qual = by(c => c.quality?.overall || 0);
        const mem = passed.slice().sort((a, b) => (a.peak_vram_gb || 0) - (b.peak_vram_gb || 0))[0];
        const tip = (c) => {
            const k = c.config || {};
            return `${c.context_length} ctx · flash ${k.flash_attention ? 'ON' : 'OFF'} · KV ${k.offload_kv_cache_to_gpu ? 'GPU' : 'CPU'} · batch ${k.eval_batch_size || 'auto'} · parallel ${k.parallel || 'auto'} · quality ${c.quality?.overall?.toFixed(3) ?? '—'}`;
        };
        const row = (label, c) => c ? `<div class="text-sm"><button class="alt-jump hover:underline" data-jump="${c.id}" title="${tip(c)}"><span class="text-gray-500">${label}:</span> <span class="font-mono">${c.context_length} ctx, ${(c.avg_generation_tok_s || 0).toFixed(1)} tok/s →</span></button></div>` : '';
        return `<div class="mt-4 p-3 bg-gray-50 rounded"><h4 class="font-medium mb-1">ALTERNATIVES</h4>
            ${row('Best Speed', speed)}${row('Best Context', ctx)}${row('Best Quality', qual)}${row('Best Memory', mem)}</div>`;
    },

    jumpToConfig(configId) {
        const row = document.querySelector(`tr[data-cfg-row-anchor="${configId}"]`)
            || document.querySelector(`[data-cfg-row="${configId}"]`);
        if (!row) return;
        row.scrollIntoView({ behavior: 'smooth', block: 'center' });
        row.classList.add('flash-highlight');
        setTimeout(() => row.classList.remove('flash-highlight'), 2000);
    },

    renderMetricCard(label, value, icon) {
        return `
            <div class="p-4 bg-gray-50 rounded-lg text-center">
                <div class="text-3xl mb-1">${icon}</div>
                <div class="text-2xl font-bold text-gray-900 font-mono">${value}</div>
                <div class="text-xs text-gray-500">${label}</div>
            </div>
        `;
    },

    renderBestConfigDetails(config) {
        const cfg = config.config || {};
        return `
            <div class="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-100">
                <h3 class="font-semibold text-blue-900 mb-3">Recommended Configuration</h3>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <div><span class="text-gray-500">Context</span><br><span class="font-mono font-medium">${config.context_length}</span></div>
                    <div><span class="text-gray-500">GPU Ratio</span><br><span class="font-mono font-medium">${cfg.gpu_ratio != null ? (cfg.gpu_ratio * 100).toFixed(0) + '%' : 'Auto'}</span></div>
                    <div><span class="text-gray-500">Flash Attn</span><br><span class="font-mono font-medium">${cfg.flash_attention ? 'ON' : 'OFF'}</span></div>
                    <div><span class="text-gray-500">KV Cache</span><br><span class="font-mono font-medium">${cfg.offload_kv_cache_to_gpu ? 'GPU' : 'CPU'}</span></div>
                    <div><span class="text-gray-500">Batch</span><br><span class="font-mono font-medium">${cfg.eval_batch_size || 'Auto'}</span></div>
                    <div><span class="text-gray-500">VRAM</span><br><span class="font-mono font-medium">${config.peak_vram_gb ? config.peak_vram_gb.toFixed(1) + ' GB' : '—'}</span></div>
                    <div><span class="text-gray-500">Score</span><br><span class="font-mono font-medium">${config.score != null ? config.score.toFixed(3) : '—'}</span></div>
                </div>
            </div>
        `;
    },

    renderTabs() {
        return `
            <div class="card">
                <div class="tabs" id="results-tabs">
                    <button class="tab active" data-tab="charts">📈 Charts</button>
                    <button class="tab" data-tab="pareto">📊 Pareto Frontier</button>
                    <button class="tab" data-tab="comparison">⚖️ Comparison</button>
                </div>
                <div class="card-body">
                    <div id="tab-panels"></div>
                </div>
            </div>
        `;
    },

    renderChartTab() {
        return `
            <div class="space-y-4">
                <div class="flex flex-wrap gap-2">
                    <button class="chart-type-btn active" data-chart="generation-context">Gen vs Context</button>
                    <button class="chart-type-btn" data-chart="vram-context">VRAM vs Context</button>
                    <button class="chart-type-btn" data-chart="speed-gpu">Speed vs GPU</button>
                    <button class="chart-type-btn" data-chart="quality-speed">Quality vs Speed</button>
                </div>
                <div class="chart-container">
                    <canvas id="main-chart"></canvas>
                </div>
            </div>
        `;
    },

    renderParetoTab() {
        return `
            <div class="space-y-4">
                <p class="text-gray-600">Configurations on the Pareto frontier represent optimal trade-offs. No other configuration is better in all metrics.</p>
                <div class="chart-container">
                    <canvas id="pareto-chart"></canvas>
                </div>
                <div class="mt-4">
                    <h4 class="font-medium mb-2">Pareto Configurations</div>
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3" id="pareto-cards">
                        ${this.state.paretoConfigs.map(c => this.renderParetoCard(c)).join('')}
                    </div>
                </div>
            </div>
        `;
    },

    renderParetoCard(config) {
        return `
            <div class="p-4 border border-gray-200 rounded-lg hover:border-blue-300 transition-colors">
                <div class="flex items-center justify-between mb-2">
                    <span class="font-mono text-sm">${config.context_length} ctx</span>
                    <span class="badge badge-success">Pareto</span>
                </div>
                <div class="grid grid-cols-3 gap-2 text-sm">
                    <div class="text-gray-500">Gen</div>
                    <div class="font-mono">${config.avg_generation_tok_s?.toFixed(1)}</div>
                    <div class="text-gray-500">tok/s</div>
                    <div class="text-gray-500">VRAM</div>
                    <div class="font-mono">${config.peak_vram_gb?.toFixed(1)} GB</div>
                    <div class="text-gray-500">Quality</div>
                    <div class="font-mono">${config.quality?.overall?.toFixed(3)}</div>
                </div>
                <div class="mt-2 text-xs text-gray-500">
                    GPU: ${config.gpu_ratio ? (config.gpu_ratio * 100).toFixed(0) + '%' : 'Auto'} | 
                    KV: ${config.offload_kv_cache_to_gpu ? 'GPU' : 'CPU'} | 
                    Flash: ${config.flash_attention ? 'ON' : 'OFF'}
                </div>
            </div>
        `;
    },

    renderComparisonTab() {
        return `
            <div class="p-4">
                <p class="text-gray-600 mb-4">Compare selected configurations side by side.</p>
                <div class="overflow-x-auto">
                    <table class="table">
                        <thead>
                            <tr>
                                <th><input type="checkbox" id="select-all-configs"></th>
                                <th>Context</th>
                                <th>GPU</th>
                                <th>Flash</th>
                                <th>KV</th>
                                <th>Batch</th>
                                <th>Gen tok/s</th>
                                <th>Prompt tok/s</th>
                                <th>TTFT</th>
                                <th>VRAM</th>
                                <th>Quality</th>
                                <th>Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${this.state.configurations.map(c => `
                                <tr>
                                    <td><input type="checkbox" class="config-checkbox" value="${c.id}"></td>
                                    <td class="font-mono">${c.context_length}</td>
                                    <td>${c.gpu_ratio ? (c.gpu_ratio * 100).toFixed(0) + '%' : 'Auto'}</td>
                                    <td>${c.flash_attention ? 'ON' : 'OFF'}</td>
                                    <td>${c.offload_kv_cache_to_gpu ? 'GPU' : 'CPU'}</td>
                                    <td>${c.eval_batch_size || 'Auto'}</td>
                                    <td class="font-mono font-medium">${c.avg_generation_tok_s?.toFixed(1)}</td>
                                    <td class="font-mono">${c.avg_prompt_tok_s?.toFixed(0)}</td>
                                    <td class="font-mono">${c.avg_ttft_ms?.toFixed(0)} ms</td>
                                    <td class="font-mono">${c.peak_vram_gb?.toFixed(1) + ' GB' || '—'}</td>
                                    <td>${c.quality?.overall?.toFixed(3) || '—'}</td>
                                    <td class="font-mono font-medium">${c.score != null ? c.score.toFixed(3) : '—'}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
                <div class="mt-4 flex gap-2">
                    <button id="compare-selected" class="btn btn-primary" disabled>Compare Selected</button>
                    <button id="clear-selection" class="btn btn-outline">Clear</button>
                </div>
            </div>
        `;
    },

    renderConfigTable() {
        return `
            <div class="card">
                <div class="card-header flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                    <div class="flex items-center gap-3">
                        <h3 class="font-semibold text-gray-900">All Tested Configurations</h3>
                        <p class="text-xs text-gray-500">Debug reveals the exact prompt, reasoning trace and model output of every test.</p>
                    </div>
                    <div class="flex items-center gap-2">
                        <button id="ask-judge-btn" class="btn btn-primary" disabled title="Select configurations or leave empty to evaluate all">
                            <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path></svg>
                            Ask Judge
                        </button>
                        <button id="judge-history-btn" class="btn btn-outline" title="View judge history">
                            <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.798 1.141a.999.999 0 001.396-.371.999.999 0 00-.371-1.396A7.04 7.04 0 0012 4c-3.866 0-7 2.889-7 6.5 0 3.61 2.834 6.5 6.5 6.5 1.15 0 2.218-.3 3.17-.843l.025-.013a.997.997 0 00-.51.048A8.99 8.99 0 0112 20c-4.963 0-9-3.577-9-8s3.577-8 8-8z"></path></svg>
                            History
                        </button>
                    </div>
                </div>
                <div class="card-body p-0">
                    <div class="table-container">
                        <table class="table">
                            <thead>
                                <tr>
                                    <th class="w-10 text-center"><input type="checkbox" id="select-all-judge-configs" title="Select/deselect all visible configurations"></th>
                                    <th>Context</th>
                                    <th>GPU</th>
                                    <th>Flash</th>
                                    <th>KV</th>
                                    <th>Batch</th>
                                    <th>Gen tok/s</th>
                                    <th>Prompt tok/s</th>
                                    <th>TTFT</th>
                                    <th>VRAM</th>
                                    <th>Quality</th>
                                    <th>Score</th>
                                    <th>Status</th>
                                    <th>Debug</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${this.state.configurations.map(c => {
                                    const cfg = c.config || {};
                                    return `
                                    <tr data-cfg-row-anchor="${c.id}" class="${c.status === 'passed' ? '' : 'bg-red-50'}">
                                        <td class="text-center"><input type="checkbox" class="judge-config-checkbox" value="${c.id}"></td>
                                        <td class="font-mono">${c.context_length}</td>
                                        <td>${cfg.gpu_ratio != null ? (cfg.gpu_ratio * 100).toFixed(0) + '%' : 'Auto'}</td>
                                        <td>${cfg.flash_attention ? 'ON' : 'OFF'}</td>
                                        <td>${cfg.offload_kv_cache_to_gpu ? 'GPU' : 'CPU'}</td>
                                        <td>${cfg.eval_batch_size || 'Auto'}</td>
                                        <td class="font-mono font-medium">${c.avg_generation_tok_s?.toFixed(1) || '—'}</td>
                                        <td class="font-mono">${c.avg_prompt_tok_s?.toFixed(0) || '—'}</td>
                                        <td class="font-mono">${c.avg_ttft_ms?.toFixed(0) || '—'} ms</td>
                                        <td class="font-mono">${c.peak_vram_gb?.toFixed(1) || '—'} GB</td>
                                        <td>${c.quality?.overall?.toFixed(3) || '—'}</td>
                                        <td class="font-mono font-medium">${c.score != null ? c.score.toFixed(3) : '—'}</td>
                                        <td><span class="badge ${this.getStatusBadge(c.status)}">${c.status}</span></td>
                                        <td><button class="btn btn-outline btn-sm cfg-detail-btn" data-cfg="${c.id}" title="Show prompt, reasoning and model output for every test of this config">Debug</button></td>
                                    </tr>
                                    <tr class="cfg-detail-row hidden" data-cfg-row="${c.id}">
                                        <td colspan="14">
                                            <div class="p-3 space-y-2">
                                                ${this.renderConfigFacts(c)}
                                                <div><a class="underline text-blue-600 text-sm" target="_blank" rel="noopener" href="/results/${this.state.runId}/configs/${c.id}">Open full JSON</a></div>
                                                <div><span class="font-medium">Per-test outputs:</span>
                                                ${(c.metrics || []).map(m => `
                                                    <div class="mt-1 border-t pt-1">
                                                        <span class="font-mono text-xs font-medium">${m.test_name}</span>
                                                        <span class="text-xs text-gray-500">tok/s ${m.generation_tok_s != null ? m.generation_tok_s.toFixed(1) : '—'} · in ${m.prompt_tokens ?? '—'} / out ${m.completion_tokens ?? '—'} tok · ${m.success ? 'ok' : 'fail: ' + (m.error || '')}</span>
                                                        <div class="text-xs font-medium mt-1">Prompt</div>
                                                        <pre class="font-mono text-xs bg-gray-50 p-2 rounded overflow-auto max-h-40">${String(m.prompt || '(prompt not stored for this run)').slice(0, 4000)}</pre>
                                                        <div class="text-xs font-medium mt-1">Thinking</div>
                                                        <pre class="font-mono text-xs bg-gray-50 p-2 rounded overflow-auto max-h-40">${String(m.thinking_text || 'n/a (old run or non-reasoning model)').slice(0, 4000)}</pre>
                                                        <div class="text-xs font-medium mt-1">Output</div>
                                                        <pre class="font-mono text-xs bg-gray-50 p-2 rounded overflow-auto max-h-40">${String(m.output_text || '—').slice(0, 2000)}</pre>
                                                        <button class="btn btn-outline btn-sm cfg-full-output-btn" data-cfg="${c.id}" data-test="${m.test_name}">Full output + quality</button>
                                                        <div class="cfg-full-output" data-full-out="${c.id}:${m.test_name}"></div>
                                                    </div>`).join('')}
                                                </div>
                                            </div>
                                        </td>
                                    </tr>
                                `;}).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    },

    renderConfigFacts(c) {
        const k = c.config || {};
        const g = c.generation || {};
        const temps = g.temperature_overrides || {};
        const fact = (label, val) => `<div><span class="text-gray-500">${label}</span><br><span class="font-mono font-medium">${val}</span></div>`;
        return `<div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            ${fact('Context', c.context_length)}
            ${fact('Flash attention', k.flash_attention ? 'ON' : 'OFF')}
            ${fact('KV cache', k.offload_kv_cache_to_gpu ? 'GPU' : 'CPU')}
            ${fact('Eval batch', k.eval_batch_size || 'server default')}
            ${fact('Physical batch', k.physical_batch_size || 'server default')}
            ${fact('Parallel', k.parallel || 'server default')}
            ${fact('Checkpoints', k.context_checkpoints ?? 'server default')}
            ${fact('Style', g.style || '—')}
            ${fact('Reasoning', g.reasoning ?? '—')}
            ${fact('Temperatures', Object.keys(temps).length ? Object.entries(temps).map(([t, v]) => `${t}=${v}`).join(', ') : 'suite defaults')}
            ${fact('top_p / top_k', 'server defaults (not tuned)')}
            ${fact('Load channel', g.load_channel || '—')}
        </div>`;
    },

    getStatusBadge(status) {
        const badges = {
            'passed': 'badge-success',
            'success': 'badge-success',
            'completed': 'badge-success',
            'completed_no_validated_winner': 'badge-warning',
            'partial_success': 'badge-warning',
            'failed': 'badge-error',
            'load_failed': 'badge-error',
            'verify_failed': 'badge-error',
            'preheat_failed': 'badge-error',
            'benchmark_failed': 'badge-error',
            'incompatible': 'badge-error',
            'quality_failed': 'badge-warning',
            'oom': 'badge-warning',
            'timeout': 'badge-warning',
            'skipped': 'badge-gray',
        };
        return badges[status] || 'badge-gray';
    },

    // ==================== JUDGE UI ====================

    openJudgeModal() {
        UI.openModal(this.renderJudgeModal(), '🤖 Ask Judge');
        this.attachJudgeModalEvents();
    },

    renderJudgeModal() {
        const models = this.state.availableModels || [];
        const passedConfigs = this.state.configurations.filter(c => c.status === 'passed');
        const selectedCount = document.querySelectorAll('.judge-config-checkbox:checked').length;
        const totalPassed = passedConfigs.length;
        const evaluateAll = selectedCount === 0;
        
        return `
            <div class="space-y-4 max-h-[70vh] overflow-y-auto">
                <div class="p-3 bg-blue-50 rounded border border-blue-200">
                    <p class="text-sm text-blue-800">
                        <strong>Evaluation scope:</strong> ${evaluateAll 
                            ? `All ${totalPassed} passed configurations (no selection = evaluate all)` 
                            : `${selectedCount} selected configuration(s) of ${totalPassed} passed`}
                    </p>
                    <p class="text-xs text-blue-700 mt-1">The judge will analyze speed, quality, stability and provide per-model winners, overall winner, task-specific guidance, and avoid-list.</p>
                </div>

                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Use Case</label>
                        <select id="judge-use-case" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <option value="general">General (balanced)</option>
                            <option value="coding">Coding</option>
                            <option value="chat">Chat / Instruction Following</option>
                            <option value="reasoning">Reasoning / Math</option>
                            <option value="structured">Structured Output (JSON)</option>
                            <option value="custom">Custom Prompt</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Priority</label>
                        <div class="flex gap-2">
                            <label class="flex-1 flex items-center gap-1"><input type="radio" name="judge-priority" value="speed" class="mr-1"> Speed</label>
                            <label class="flex-1 flex items-center gap-1"><input type="radio" name="judge-priority" value="quality" class="mr-1"> Quality</label>
                            <label class="flex-1 flex items-center gap-1"><input type="radio" name="judge-priority" value="balanced" checked class="mr-1"> Balanced</label>
                            <label class="flex-1 flex items-center gap-1"><input type="radio" name="judge-priority" value="context" class="mr-1"> Context</label>
                        </div>
                    </div>
                </div>

                <div id="judge-custom-prompt-container" class="hidden">
                    <label class="block text-sm font-medium text-gray-700 mb-1">Custom Prompt</label>
                    <textarea id="judge-custom-prompt" rows="3" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="Describe your specific evaluation criteria..."></textarea>
                </div>

                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Backend</label>
                        <div class="flex gap-2">
                            <label class="flex-1 flex items-center gap-1"><input type="radio" name="judge-backend" value="heuristic" checked class="mr-1"> Heuristic (fast, no AI)</label>
                            <label class="flex-1 flex items-center gap-1"><input type="radio" name="judge-backend" value="local" class="mr-1"> Local (LM Studio)</label>
                            <label class="flex-1 flex items-center gap-1"><input type="radio" name="judge-backend" value="api" class="mr-1"> API (OpenRouter)</label>
                        </div>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Judge Model</label>
                        <select id="judge-model-select" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <option value="heuristic" selected>Heuristic (no model)</option>
                            ${models.filter(m => !m.id.startsWith('text-embedding')).map(m => 
                                `<option value="${m.id}">${m.id} (${m.size_gb?.toFixed(1) || '?'} GB)</option>`
                            ).join('')}
                        </select>
                    </div>
                </div>

                <div id="judge-api-settings" class="hidden space-y-3">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">API Base URL</label>
                        <input type="text" id="judge-api-base" value="https://openrouter.ai/api/v1" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="https://openrouter.ai/api/v1">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">API Key</label>
                        <input type="password" id="judge-api-key" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="sk-...">
                    </div>
                </div>

                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Top-N Detail <span id="judge-top-n-label">3</span></label>
                        <input type="range" id="judge-top-n" min="1" max="10" value="3" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Options</label>
                        <div class="flex flex-col gap-2 mt-1">
                            <label class="flex items-center gap-2"><input type="checkbox" id="judge-include-reasoning" checked> Include reasoning matrix (off/low/medium/high/xhigh)</label>
                            <label class="flex items-center gap-2"><input type="checkbox" id="judge-fastest-context"> Fastest @ max context preset</label>
                        </div>
                    </div>
                </div>

                <div class="flex justify-end gap-2 pt-4 border-t border-gray-200">
                    <button type="button" class="btn btn-outline" onclick="UI.closeModal()">Cancel</button>
                    <button type="button" id="run-judge-btn" class="btn btn-primary">
                        <svg class="w-4 h-4 mr-1 animate-spin hidden" id="judge-spinner" fill="none" stroke="currentColor" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                        <span>Run Judge</span>
                    </button>
                </div>
            </div>
        `;
    },

    renderVerdictModal(verdict, metadata = {}) {
        const confidence = verdict.confidence || 0;
        const confidenceColor = confidence >= 0.8 ? 'text-green-600' : confidence >= 0.6 ? 'text-yellow-600' : 'text-red-600';
        const confidenceLabel = confidence >= 0.8 ? 'High' : confidence >= 0.6 ? 'Medium' : 'Low';
        
        const formatTask = (task) => {
            if (!task) return '<span class="text-gray-500">—</span>';
            return `<strong>${task.winner || '—'}</strong> ${task.runner_up ? `<span class="text-gray-500 ml-1">(runner-up: ${task.runner_up})</span>` : ''} ${task.reasoning ? `<p class="text-xs text-gray-500 mt-1">${task.reasoning}</p>` : ''}`;
        };

        return `
            <div class="space-y-4 max-h-[80vh] overflow-y-auto">
                <div class="flex items-center justify-between">
                    <h3 class="text-lg font-semibold">Judge Verdict</h3>
                    <span class="px-2 py-1 rounded-full text-xs font-medium ${confidenceColor} bg-opacity-10">
                        Confidence: ${(confidence * 100).toFixed(0)}% (${confidenceLabel})
                    </span>
                </div>

                <div class="tabs tabs-border" id="verdict-tabs">
                    <button class="tab active" data-verdict-tab="overall">Overall</button>
                    <button class="tab" data-verdict-tab="per-model">Per-Model</button>
                    <button class="tab" data-verdict-tab="by-task">By Task</button>
                    <button class="tab" data-verdict-tab="avoid">Avoid</button>
                    <button class="tab" data-verdict-tab="raw">Raw JSON</button>
                </div>

                <div id="verdict-tab-panels" class="space-y-4">
                    <!-- Overall -->
                    <div id="verdict-panel-overall" class="verdict-panel">
                        <div class="card">
                            <div class="card-header"><h4 class="font-semibold">Overall Winner</h4></div>
                            <div class="card-body">
                                ${verdict.overall_winner ? `
                                    <p><strong>Model:</strong> ${verdict.overall_winner.model_id || '—'}</p>
                                    <p><strong>Config:</strong> ${verdict.overall_winner.config_id || '—'}</p>
                                    <p><strong>Reasoning:</strong> ${verdict.overall_winner.reasoning || '—'}</p>
                                ` : '<p class="text-gray-500">No overall winner determined.</p>'}
                            </div>
                        </div>
                        ${verdict.per_model_winner ? `
                        <div class="card mt-4">
                            <div class="card-header"><h4 class="font-semibold">Per-Model Winners</h4></div>
                            <div class="card-body">
                                <p><strong>Model:</strong> ${verdict.per_model_winner.model_id || '—'}</p>
                                <p><strong>Config:</strong> ${verdict.per_model_winner.config_id || '—'}</p>
                                <p><strong>Reasoning:</strong> ${verdict.per_model_winner.reasoning || '—'}</p>
                            </div>
                        </div>` : ''}
                        <div class="card mt-4">
                            <div class="card-header"><h4 class="font-semibold">Metadata</h4></div>
                            <div class="card-body">
                                <p><strong>Backend:</strong> ${metadata.backend || '—'}</p>
                                <p><strong>Judge Model:</strong> ${metadata.judge_model || '—'}</p>
                                <p><strong>Use Case:</strong> ${metadata.use_case || '—'}</p>
                                <p><strong>Priority:</strong> ${metadata.priority || '—'}</p>
                                <p><strong>Configs Evaluated:</strong> ${metadata.configs_evaluated || '—'}</p>
                                <p><strong>Latency:</strong> ${metadata.latency_ms ? metadata.latency_ms + ' ms' : '—'}</p>
                                <p><strong>Created:</strong> ${metadata.created_at || '—'}</p>
                            </div>
                        </div>
                    </div>

                    <!-- Per-Model -->
                    <div id="verdict-panel-per-model" class="verdict-panel hidden">
                        ${verdict.per_model_winner ? `
                        <div class="card">
                            <div class="card-header"><h4 class="font-semibold">Per-Model Winner</h4></div>
                            <div class="card-body">
                                <p><strong>Model:</strong> ${verdict.per_model_winner.model_id || '—'}</p>
                                <p><strong>Config:</strong> ${verdict.per_model_winner.config_id || '—'}</p>
                                <p><strong>Reasoning:</strong> ${verdict.per_model_winner.reasoning || '—'}</p>
                            </div>
                        </div>` : '<p class="text-gray-500">No per-model winner data.</p>'}
                    </div>

                    <!-- By Task -->
                    <div id="verdict-panel-by-task" class="verdict-panel hidden">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div class="card"><div class="card-header"><h4 class="font-semibold">Coding</h4></div><div class="card-body">${formatTask(verdict.task_breakdown?.coding)}</div></div>
                            <div class="card"><div class="card-header"><h4 class="font-semibold">Chat</h4></div><div class="card-body">${formatTask(verdict.task_breakdown?.chat)}</div></div>
                            <div class="card"><div class="card-header"><h4 class="font-semibold">Reasoning</h4></div><div class="card-body">${formatTask(verdict.task_breakdown?.reasoning)}</div></div>
                            <div class="card"><div class="card-header"><h4 class="font-semibold">Structured Output</h4></div><div class="card-body">${formatTask(verdict.task_breakdown?.structured)}</div></div>
                        </div>
                    </div>

                    <!-- Avoid -->
                    <div id="verdict-panel-avoid" class="verdict-panel hidden">
                        <div class="card">
                            <div class="card-header"><h4 class="font-semibold text-red-600">Models/Configs to Avoid</h4></div>
                            <div class="card-body">
                                ${(verdict.avoid || []).length ? `
                                    <ul class="space-y-2">
                                        ${(verdict.avoid || []).map(a => `
                                            <li class="flex items-start gap-2 p-2 bg-red-50 rounded">
                                                <svg class="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293a1 1 0 00-1.414 0z" clip-rule="evenodd"></path></svg>
                                                <div>
                                                    <p class="font-medium text-red-800">${a.model_id || '—'}</p>
                                                    <p class="text-sm text-red-600">${a.reason || '—'}</p>
                                                </div>
                                            </li>`).join('')}
                                    </ul>
                                ` : '<p class="text-gray-500">No avoid recommendations.'}</p>
                                ${(verdict.warnings || []).length ? `
                                    <h5 class="font-medium mt-4 mb-2 text-amber-700">Warnings</h5>
                                    <ul class="space-y-1">
                                        ${(verdict.warnings || []).map(w => `<li class="text-sm text-amber-700 flex items-center gap-1"><svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path></svg>${w}</li>`).join('')}
                                    </ul>
                                ` : ''}
                            </div>
                        </div>
                    </div>

                    <!-- Raw JSON -->
                    <div id="verdict-panel-raw" class="verdict-panel hidden">
                        <div class="card">
                            <div class="card-header flex justify-between">
                                <h4 class="font-semibold">Raw Verdict JSON</h4>
                                <button class="btn btn-outline btn-sm" onclick="navigator.clipboard.writeText(JSON.stringify(${JSON.stringify(verdict).replace(/"/g, '\\"')}, null, 2))">Copy JSON</button>
                            </div>
                            <div class="card-body">
                                <pre class="font-mono text-xs bg-gray-50 p-4 rounded overflow-auto max-h-96">${JSON.stringify(verdict, null, 2)}</pre>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    },

    renderHistoryModal(history) {
        return `
            <div class="space-y-4 max-h-[70vh] overflow-y-auto">
                <p class="text-sm text-gray-600">${history.length} recent judge evaluation(s). Click a row to view details.</p>
                <div class="overflow-x-auto">
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Created</th>
                                <th>Use Case</th>
                                <th>Backend</th>
                                <th>Judge Model</th>
                                <th>Confidence</th>
                                <th>Overall Winner</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${history.map(h => `
                                <tr class="cursor-pointer hover:bg-gray-50" data-history-id="${h.id}">
                                    <td class="text-xs font-mono">${h.created_at || '—'}</td>
                                    <td>${h.use_case || '—'}</td>
                                    <td>${h.backend || '—'}</td>
                                    <td class="font-mono text-xs">${h.judge_model || '—'}</td>
                                    <td class="font-mono">${h.confidence != null ? (h.confidence * 100).toFixed(0) + '%' : '—'}</td>
                                    <td class="font-mono text-xs">${h.overall_winner_model || '—'}</td>
                                    <td><span class="badge ${h.status === 'completed' ? 'badge-success' : h.status === 'failed' ? 'badge-error' : 'badge-warning'}">${h.status || '—'}</span></td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
                ${history.length === 0 ? '<p class="text-center text-gray-500 py-8">No judge evaluations yet.</p>' : ''}
            </div>
        `;
    },

    async loadAvailableModels() {
        try {
            const resp = await fetch('/api/models');
            if (resp.ok) {
                const data = await resp.json();
                this.state.availableModels = data.data || data;
            }
        } catch (e) {
            console.warn('Failed to load models for judge modal:', e);
            this.state.availableModels = [];
        }
    },

    attachJudgeModalEvents() {
        // Use case change
        const useCaseSelect = document.getElementById('judge-use-case');
        if (useCaseSelect) {
            useCaseSelect.addEventListener('change', (e) => {
                document.getElementById('judge-custom-prompt-container').classList.toggle('hidden', e.target.value !== 'custom');
            });
        }

        // Backend change
        document.querySelectorAll('input[name="judge-backend"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                const isApi = e.target.value === 'api';
                document.getElementById('judge-api-settings').classList.toggle('hidden', !isApi);
                const modelSelect = document.getElementById('judge-model-select');
                if (isApi) {
                    // For API, allow any model name
                    modelSelect.disabled = false;
                    if (modelSelect.value === 'heuristic') {
                        modelSelect.value = 'anthropic/claude-3.5-sonnet';
                    }
                } else {
                    modelSelect.disabled = false;
                }
            });
        });

        // Top-N slider
        const topNSlider = document.getElementById('judge-top-n');
        if (topNSlider) {
            topNSlider.addEventListener('input', (e) => {
                document.getElementById('judge-top-n-label').textContent = e.target.value;
            });
        }

        // Run Judge button
        const runBtn = document.getElementById('run-judge-btn');
        if (runBtn) {
            runBtn.addEventListener('click', () => this.runJudge());
        }

        // Verdict tabs
        document.querySelectorAll('#verdict-tabs .tab').forEach(tab => {
            tab.addEventListener('click', () => this.switchVerdictTab(tab.dataset.verdictTab));
        });

        // History table clicks
        document.querySelectorAll('[data-history-id]').forEach(row => {
            row.addEventListener('click', () => this.openVerdictModalFromHistory(row.dataset.historyId));
        });
    },

    switchVerdictTab(tabName) {
        document.querySelectorAll('#verdict-tabs .tab').forEach(t => {
            t.classList.toggle('active', t.dataset.verdictTab === tabName);
        });
        document.querySelectorAll('.verdict-panel').forEach(p => {
            p.classList.toggle('hidden', p.id !== `verdict-panel-${tabName}`);
        });
    },

    async runJudge() {
        const runBtn = document.getElementById('run-judge-btn');
        const spinner = document.getElementById('judge-spinner');
        const btnText = runBtn.querySelector('span');
        
        runBtn.disabled = true;
        spinner.classList.remove('hidden');
        btnText.textContent = 'Evaluating...';
        UI.showToast('Judge evaluation started...', 'info');

        const runIds = [this.state.runId];
        const configIds = Array.from(document.querySelectorAll('.judge-config-checkbox:checked')).map(cb => cb.value);
        const evaluateAll = configIds.length === 0;

        try {
            const request = {
                run_ids: runIds,
                config_ids: evaluateAll ? undefined : configIds,
                use_case: document.getElementById('judge-use-case').value,
                custom_prompt: document.getElementById('judge-use-case').value === 'custom' ? document.getElementById('judge-custom-prompt').value : null,
                priority: document.querySelector('input[name="judge-priority"]:checked')?.value || 'balanced',
                backend: document.querySelector('input[name="judge-backend"]:checked')?.value || 'heuristic',
                judge_model: document.getElementById('judge-model-select').value,
                api_key: document.getElementById('judge-api-key').value || null,
                api_base: document.getElementById('judge-api-base').value || null,
                top_n_detail: parseInt(document.getElementById('judge-top-n').value, 10),
                include_reasoning_matrix: document.getElementById('judge-include-reasoning').checked,
                fastest_context_preset: document.getElementById('judge-fastest-context').checked,
                wait: false,
            };

            const resp = await fetch('/api/judge', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(request),
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Judge request failed');
            }

            const data = await resp.json();
            
            if (request.wait || data.status === 'completed') {
                UI.closeModal();
                this.showVerdictModal(data.verdict, data);
            } else {
                // Poll for completion
                this.pollJudgeResult(data.judge_id);
            }
        } catch (error) {
            UI.showToast('Judge failed: ' + error.message, 'error');
        } finally {
            runBtn.disabled = false;
            document.getElementById('judge-spinner').classList.add('hidden');
            btnText.textContent = 'Run Judge';
        }
    },

    async pollJudgeResult(judgeId) {
        const poll = async () => {
            try {
                const resp = await fetch(`/api/judge/${judgeId}`);
                if (!resp.ok) throw new Error('Poll failed');
                const data = await resp.json();
                
                if (data.status === 'completed') {
                    UI.closeModal();
                    this.showVerdictModal(data.verdict, data);
                    UI.showToast('Judge evaluation complete', 'success');
                } else if (data.status === 'failed') {
                    UI.showToast('Judge failed: ' + data.error_message, 'error');
                } else {
                    // Still pending, poll again in 2 seconds
                    setTimeout(poll, 2000);
                }
            } catch (e) {
                console.error('Poll error:', e);
                setTimeout(poll, 5000);
            }
        };
        setTimeout(poll, 2000);
    },

    showVerdictModal(verdict, metadata) {
        UI.openModal(this.renderVerdictModal(verdict, metadata), '🤖 Judge Verdict');
        // Attach verdict tab events
        setTimeout(() => {
            document.querySelectorAll('#verdict-tabs .tab').forEach(tab => {
                tab.addEventListener('click', () => this.switchVerdictTab(tab.dataset.verdictTab));
            });
        }, 0);
    },

    async openJudgeHistory() {
        try {
            const resp = await fetch('/api/judge/history?limit=50');
            if (!resp.ok) throw new Error('Failed to load history');
            const history = await resp.json();
            UI.openModal(this.renderHistoryModal(history), '📜 Judge History');
            // Attach click events for history rows
            setTimeout(() => {
                document.querySelectorAll('[data-history-id]').forEach(row => {
                    row.addEventListener('click', () => this.openVerdictModalFromHistory(row.dataset.historyId));
                });
            }, 0);
        } catch (e) {
            UI.showToast('Failed to load judge history: ' + e.message, 'error');
        }
    },

    async openVerdictModalFromHistory(judgeId) {
        try {
            const resp = await fetch(`/api/judge/${judgeId}`);
            if (!resp.ok) throw new Error('Failed to load verdict');
            const data = await resp.json();
            UI.closeModal();
            this.showVerdictModal(data.verdict, data);
        } catch (e) {
            UI.showToast('Failed to load verdict: ' + e.message, 'error');
        }
    },

    switchVerdictTab(tabName) {
        document.querySelectorAll('#verdict-tabs .tab').forEach(t => {
            t.classList.toggle('active', t.dataset.verdictTab === tabName);
        });
        document.querySelectorAll('.verdict-panel').forEach(p => {
            p.classList.toggle('hidden', p.id !== `verdict-panel-${tabName}`);
        });
    },

    attachEvents() {
        // Tab switching
        document.querySelectorAll('#results-tabs .tab').forEach(tab => {
            tab.addEventListener('click', () => this.switchTab(tab.dataset.tab));
        });

        // Chart type buttons
        document.querySelectorAll('.chart-type-btn').forEach(btn => {
            btn.addEventListener('click', () => this.switchChart(btn.dataset.chart));
        });

        // Pause/Resume buttons
        const pauseBtn = document.getElementById('pause-run-btn');
        if (pauseBtn) {
            pauseBtn.addEventListener('click', () => this.pauseRun());
        }
        const resumeBtn = document.getElementById('resume-run-btn');
        if (resumeBtn) {
            resumeBtn.addEventListener('click', () => this.resumeRun());
        }

        // Apply best config
        const applyBtn = document.getElementById('apply-best-btn');
        if (applyBtn) {
            applyBtn.addEventListener('click', () => this.applyBestConfig());
        }

        // Export
        const exportBtn = document.getElementById('export-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportResults());
        }

        // Select all configs
        const selectAll = document.getElementById('select-all-configs');
        if (selectAll) {
            selectAll.addEventListener('change', (e) => {
                document.querySelectorAll('.config-checkbox').forEach(cb => {
                    cb.checked = e.target.checked;
                });
                this.updateCompareButton();
            });
        }

        // Individual checkboxes
        document.querySelectorAll('.config-checkbox').forEach(cb => {
            cb.addEventListener('change', () => this.updateCompareButton());
        });

        // Compare selected
        const compareBtn = document.getElementById('compare-selected');
        if (compareBtn) {
            compareBtn.addEventListener('click', () => this.compareSelected());
        }

        // Clear selection
        const clearBtn = document.getElementById('clear-selection');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                document.querySelectorAll('.config-checkbox').forEach(cb => cb.checked = false);
                document.getElementById('select-all-configs').checked = false;
                this.updateCompareButton();
            });
        }

        // Alternatives jump links
        document.querySelectorAll('.alt-jump').forEach(btn => {
            btn.addEventListener('click', () => this.jumpToConfig(btn.dataset.jump));
        });

        // Expandable config detail rows
        document.querySelectorAll('.cfg-detail-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const row = document.querySelector(`[data-cfg-row="${btn.dataset.cfg}"]`);
                if (row) row.classList.toggle('hidden');
            });
        });

    },

    async loadGenerationProfiles() {
        const box = document.getElementById('gen-profiles');
        if (!box) return;
        const model = this.state.run?.model;
        if (!model) return;
        try {
            const r = await fetch(`/api/generation-profiles?model_id=${encodeURIComponent(model.id)}`);
            if (!r.ok) throw new Error('HTTP ' + r.status);
            const j = await r.json();
            const fmt = (v) => v == null ? 'server default' : String(v);
            const col = (title, d) => d
                ? `<div><span class="font-medium">${title}</span> <span class="text-xs text-gray-500">(${d.source})</span><br>`
                  + `<span class="font-mono text-xs">temp ${fmt(d.temperature)} · top_p ${fmt(d.top_p)} · top_k ${fmt(d.top_k)}</span><br>`
                  + `<span class="text-xs text-gray-500">${d.note || ''}</span></div>`
                : `<div><span class="font-medium">${title}</span><br><span class="text-xs text-gray-500">check the model card</span></div>`;
            box.innerHTML = '<h4 class="font-medium mb-1">GENERATION PROFILES</h4>' + Object.entries(j.profiles || {}).map(([name, p]) =>
                `<div class="mt-2 border-t pt-1"><span class="font-medium capitalize">${name}</span> <span class="text-xs text-gray-500">— ${p.use || ''}</span>`
                + `<div class="grid md:grid-cols-2 gap-2 mt-1">${col('Measured (benchmark)', p.ours)}${col('Publisher', p.publisher)}</div></div>`
            ).join('');
        } catch (e) {
            box.innerHTML = '<h4 class="font-medium mb-1">GENERATION PROFILES</h4>'
                + `<p class="text-xs text-gray-500">Unavailable: ${String((e && e.message) || e).slice(0, 160)}</p>`;
        }
    },

    async loadFullOutput(btn) {
        const slot = document.querySelector(
            `[data-full-out="${btn.dataset.cfg}:${btn.dataset.test}"]`
        );
        if (!slot) return;
        if (slot.dataset.loaded) {
            slot.classList.toggle('hidden');
            return;
        }
        slot.innerHTML = '<p class="text-xs text-gray-500">Loading full output...</p>';
        try {
            const detail = await API.getConfiguration(this.state.runId, btn.dataset.cfg);
            const m = (detail.metrics || []).find(x => x.test_name === btn.dataset.test);
            if (!m) throw new Error('test not found');
            const q = m.quality_overall != null ? `quality ${m.quality_overall}` : 'quality n/a';
            const warn = m.thinking_outside_json
                ? '<p class="text-xs mt-1"><span class="badge badge-warning" title="Model reasoned outside the JSON object; structure extracted with format penalty">⚠ thinking outside JSON — passed with warning</span></p>'
                : '';
            slot.innerHTML = `<p class="text-xs text-gray-500">${q} · in ${m.prompt_tokens ?? '—'} / out ${m.completion_tokens ?? '—'} tok</p>${warn}`
                + `<div class="text-xs font-medium mt-1">Thinking</div>`
                + `<pre class="font-mono text-xs bg-gray-50 p-2 rounded overflow-auto max-h-96">${String(m.thinking_text || 'n/a').slice(0, 20000)}</pre>`
                + `<div class="text-xs font-medium mt-1">Output</div>`
                + `<pre class="font-mono text-xs bg-gray-50 p-2 rounded overflow-auto max-h-96">${String(m.output_text || '—').slice(0, 20000)}</pre>`;
            slot.dataset.loaded = '1';
        } catch (e) {
            slot.innerHTML = `<p class="text-xs text-red-600">Failed: ${String((e && e.message) || e).slice(0, 200)}</p>`;
        }
    },

    switchTab(tabName) {
        document.querySelectorAll('#results-tabs .tab').forEach(t => {
            t.classList.toggle('active', t.dataset.tab === tabName);
        });

        const panel = document.getElementById('tab-panels');
        if (tabName === 'charts') {
            panel.innerHTML = this.renderChartTab();
            this.attachChartEvents();
            this.renderChart();
        } else if (tabName === 'pareto') {
            panel.innerHTML = this.renderParetoTab();
            this.renderParetoChart();
        } else if (tabName === 'comparison') {
            panel.innerHTML = this.renderComparisonTab();
            this.attachComparisonEvents();
        }
    },

    attachChartEvents() {
        document.querySelectorAll('.chart-type-btn').forEach(btn => {
            btn.addEventListener('click', () => this.switchChart(btn.dataset.chart));
        });
    },

    attachComparisonEvents() {
        const selectAll = document.getElementById('select-all-configs');
        if (selectAll) {
            selectAll.addEventListener('change', (e) => {
                document.querySelectorAll('.config-checkbox').forEach(cb => cb.checked = e.target.checked);
                this.updateCompareButton();
            });
        }

        document.querySelectorAll('.config-checkbox').forEach(cb => {
            cb.addEventListener('change', () => this.updateCompareButton());
        });

        const compareBtn = document.getElementById('compare-selected');
        if (compareBtn) {
            compareBtn.addEventListener('click', () => this.compareSelected());
        }

        const clearBtn = document.getElementById('clear-selection');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                document.querySelectorAll('.config-checkbox').forEach(cb => cb.checked = false);
                document.getElementById('select-all-configs').checked = false;
                this.updateCompareButton();
            });
        }
    },

    updateCompareButton() {
        const selected = document.querySelectorAll('.config-checkbox:checked').length;
        const btn = document.getElementById('compare-selected');
        if (btn) {
            btn.disabled = selected < 2;
            btn.textContent = `Compare Selected (${selected})`;
        }
    },

    switchChart(chartType) {
        this.state.currentChart = chartType;
        document.querySelectorAll('.chart-type-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.chart === chartType);
        });
        this.renderChart();
    },

    renderChart() {
        const ctx = document.getElementById('main-chart');
        if (!ctx) return;
        if (typeof Chart === 'undefined') {
            ctx.parentElement.innerHTML = '<p class="text-sm text-gray-500">Charts unavailable (chart.js failed to load). Data tables below are unaffected.</p>';
            return;
        }

        const data = this.prepareChartData();
        if (!data.length) return;

        try {
            Charts.destroyChart('main-chart');
        } catch (e) {
            console.warn('Chart cleanup failed:', e);
        }
        switch (this.state.currentChart) {
            case 'generation-context':
                Charts.createGenerationVsContext(ctx.getContext('2d'), data);
                break;
            case 'vram-context':
                Charts.createVRAMVsContext(ctx.getContext('2d'), data);
                break;
            case 'speed-gpu':
                Charts.createSpeedVsGPURatio(ctx.getContext('2d'), data);
                break;
            case 'quality-speed':
                Charts.createQualityVsSpeed(ctx.getContext('2d'), data);
                break;
        }
    },

    renderParetoChart() {
        const ctx = document.getElementById('pareto-chart');
        if (!ctx) return;

        const allData = this.prepareChartData();
        const paretoData = this.state.paretoConfigs.map(c => ({
            context_length: c.context_length,
            generation_tok_s: c.avg_generation_tok_s,
            config: c,
        }));

        Charts.destroyChart('pareto-chart');
        Charts.createParetoFrontier(ctx.getContext('2d'), allData, paretoData);
    },

    prepareChartData() {
        return this.state.configurations
            .filter(c => c.status === 'passed')
            .map(c => ({
                context_length: c.context_length,
                generation_tok_s: c.avg_generation_tok_s,
                prompt_tok_s: c.avg_prompt_tok_s,
                ttft_ms: c.avg_ttft_ms,
                peak_vram_gb: c.peak_vram_gb,
                gpu_ratio: c.gpu_ratio,
                flash_attention: c.flash_attention,
                offload_kv_cache_to_gpu: c.offload_kv_cache_to_gpu,
                eval_batch_size: c.eval_batch_size,
                quality: c.quality?.overall,
                score: c.score,
                config: c,
            }));
    },

    async applyBestConfig() {
        const run = this.state.run;
        const bestConfig = run.best_config_id
            ? this.state.configurations.find(c => c.id === run.best_config_id)
            : null;

        if (!bestConfig) return;

        const ok = window.confirm(
            'Load this model in LM Studio with the winning configuration?\n\n'
            + 'One-shot load only — it does NOT change any saved default. '
            + 'After unload/restart LM Studio uses its own defaults again.'
        );
        if (!ok) return;
        UI.showLoading('Applying configuration...');
        try {
            await API.applyConfiguration({
                model_id: run.model.id,
                config: bestConfig.config,
            });
            UI.hideLoading();
            UI.showToast('Configuration applied successfully', 'success');
        } catch (error) {
            UI.hideLoading();
            UI.showToast(error.message || 'Failed to apply configuration', 'error');
        }
    },

    async exportResults() {
        try {
            const blob = await API.exportRun(this.state.runId, 'json');
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `run_${this.state.runId}.json`;
            a.click();
            URL.revokeObjectURL(url);
            UI.showToast('Results exported', 'success');
        } catch (error) {
            UI.showToast('Export failed', 'error');
        }
    },

    async pauseRun() {
        const btn = document.getElementById('pause-run-btn');
        if (btn) btn.disabled = true;
        UI.showLoading('Requesting pause...');
        try {
            const result = await API.pauseRun(this.state.runId);
            UI.hideLoading();
            UI.showToast(result.message || 'Pause requested — will take effect at next safe boundary', 'success');
            // Reload run to show updated status
            setTimeout(() => window.location.reload(), 1000);
        } catch (error) {
            UI.hideLoading();
            UI.showToast(error.message || 'Pause failed', 'error');
            if (btn) btn.disabled = false;
        }
    },

    async resumeRun() {
        const btn = document.getElementById('resume-run-btn');
        if (btn) btn.disabled = true;
        UI.showLoading('Resuming from checkpoint...');
        try {
            const result = await API.resumeRunFromCheckpoint(this.state.runId);
            UI.hideLoading();
            UI.showToast(`Resumed — ${result.completed || 0} configurations already completed`, 'success');
            // Reload to show live progress
            setTimeout(() => window.location.reload(), 1500);
        } catch (error) {
            UI.hideLoading();
            UI.showToast(error.message || 'Resume failed', 'error');
            if (btn) btn.disabled = false;
        }
    },

    compareSelected() {
        const selected = document.querySelectorAll('.config-checkbox:checked');
        if (selected.length < 2) return;

        const configs = Array.from(selected).map(cb => {
            return this.state.configurations.find(c => c.id === cb.value);
        }).filter(Boolean);

        UI.openModal(this.renderComparisonModal(configs), 'Compare Configurations');
    },

    renderComparisonModal(configs) {
        if (!configs.length) return '';

        return `
            <div class="overflow-x-auto max-h-96">
                <table class="table">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            ${configs.map(c => `
                                <th class="text-center">
                                    <div class="font-mono text-sm">${c.context_length} ctx</div>
                                    <div class="text-xs text-gray-400">GPU ${c.gpu_ratio ? (c.gpu_ratio * 100).toFixed(0) + '%' : 'Auto'}</div>
                                </th>
                            `).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${this.renderComparisonRow('Generation tok/s', configs, 'avg_generation_tok_s', '.1f')}
                        ${this.renderComparisonRow('Prompt tok/s', configs, 'avg_prompt_tok_s', '.0f')}
                        ${this.renderComparisonRow('TTFT (ms)', configs, 'avg_ttft_ms', '.0f')}
                        ${this.renderComparisonRow('VRAM (GB)', configs, 'peak_vram_gb', '.1f')}
                        ${this.renderComparisonRow('Quality', configs, 'quality.overall', '.3f')}
                        ${this.renderComparisonRow('Score', configs, 'score', '.3f')}
                    </tbody>
                </table>
            </div>
        `;
    },

    renderComparisonRow(label, configs, path, format) {
        const values = configs.map(c => {
            const val = path.split('.').reduce((obj, key) => obj?.[key], c);
            if (val == null) return '—';
            return format ? val.toFixed(format.replace('.', '')) : val;
        });

        return `
            <tr>
                <td class="font-medium text-gray-700">${label}</td>
                ${values.map(v => `<td class="text-center font-mono font-medium">${v}</td>`).join('')}
            </tr>
        `;
    },
};

// Initialize
// readyState-safe boot: deferred modules may evaluate after DOMContentLoaded.
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => ResultsPage.init());
} else {
    ResultsPage.init();
}

// Charts bake colors at creation: re-render current view on theme switch.
window.addEventListener('lm-theme-changed', () => {
    try {
        if (typeof Charts !== 'undefined' && Charts && Charts.init) Charts.init();
        const active = document.querySelector('#results-tabs .tab.active');
        if (active && ResultsPage.renderChart) ResultsPage.renderChart();
    } catch (e) { /* theme-only refresh, never fatal */ }
});