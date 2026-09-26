// LM Studio Auto Optimizer - Results Page

import { API } from './api.js?v=4';
import { UI } from './ui.js?v=4';
import { Charts } from './charts.js?v=4';
import { WebSocketManager } from './websocket.js?v=4';

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

        // ZERO-PASS: explicit failure state, no Apply offer.
        if (!bestConfig) {
            return `
            <div class="card border-red-300">
                <div class="card-header"><h1 class="text-2xl font-bold text-red-700">Optimization unsuccessful</h1>
                <p class="text-sm text-gray-500 font-mono">${run.model?.id || ''}</p></div>
                <div class="card-body">
                    <p>No valid configurations were successfully benchmarked.</p>
                    <p class="mt-2 text-sm">Reason: load_failures=${failed.length - oom - timeouts}, oom=${oom}, timeouts=${timeouts}</p>
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

    renderAlternatives(best) {
        const passed = this.state.configurations.filter(c => c.status === 'passed');
        if (!passed.length) return '';
        const by = (fn) => passed.slice().sort((a, b) => fn(b) - fn(a))[0];
        const speed = by(c => c.avg_generation_tok_s || 0);
        const ctx = by(c => c.context_length || 0);
        const qual = by(c => c.quality?.overall || 0);
        const mem = passed.slice().sort((a, b) => (a.peak_vram_gb || 0) - (b.peak_vram_gb || 0))[0];
        const row = (label, c) => c ? `<div class="text-sm"><span class="text-gray-500">${label}:</span> <span class="font-mono">${c.context_length} ctx, ${(c.avg_generation_tok_s || 0).toFixed(1)} tok/s</span></div>` : '';
        return `<div class="mt-4 p-3 bg-gray-50 rounded"><h4 class="font-medium mb-1">ALTERNATIVES</h4>
            ${row('Best Speed', speed)}${row('Best Context', ctx)}${row('Best Quality', qual)}${row('Best Memory', mem)}</div>`;
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
        return `
            <div class="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-100">
                <h3 class="font-semibold text-blue-900 mb-3">Recommended Configuration</h3>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <div><span class="text-gray-500">Context</span><br><span class="font-mono font-medium">${config.context_length}</span></div>
                    <div><span class="text-gray-500">GPU Ratio</span><br><span class="font-mono font-medium">${config.gpu_ratio ? (config.gpu_ratio * 100).toFixed(0) + '%' : 'Auto'}</span></div>
                    <div><span class="text-gray-500">Flash Attn</span><br><span class="font-mono font-medium">${config.flash_attention ? 'ON' : 'OFF'}</span></div>
                    <div><span class="text-gray-500">KV Cache</span><br><span class="font-mono font-medium">${config.offload_kv_cache_to_gpu ? 'GPU' : 'CPU'}</span></div>
                    <div><span class="text-gray-500">Batch</span><br><span class="font-mono font-medium">${config.eval_batch_size || 'Auto'}</span></div>
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
                <div class="card-header">
                    <h3 class="font-semibold text-gray-900">All Tested Configurations</h3>
                </div>
                <div class="card-body p-0">
                    <div class="table-container">
                        <table class="table">
                            <thead>
                                <tr>
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
                                </tr>
                            </thead>
                            <tbody>
                                ${this.state.configurations.map(c => `
                                    <tr class="${c.status === 'passed' ? '' : 'bg-red-50'}">
                                        <td class="font-mono">${c.context_length}</td>
                                        <td>${c.gpu_ratio ? (c.gpu_ratio * 100).toFixed(0) + '%' : 'Auto'}</td>
                                        <td>${c.flash_attention ? 'ON' : 'OFF'}</td>
                                        <td>${c.offload_kv_cache_to_gpu ? 'GPU' : 'CPU'}</td>
                                        <td>${c.eval_batch_size || 'Auto'}</td>
                                        <td class="font-mono font-medium">${c.avg_generation_tok_s?.toFixed(1) || '—'}</td>
                                        <td class="font-mono">${c.avg_prompt_tok_s?.toFixed(0) || '—'}</td>
                                        <td class="font-mono">${c.avg_ttft_ms?.toFixed(0) || '—'} ms</td>
                                        <td class="font-mono">${c.peak_vram_gb?.toFixed(1) || '—'} GB</td>
                                        <td>${c.quality?.overall?.toFixed(3) || '—'}</td>
                                        <td class="font-mono font-medium">${c.score != null ? c.score.toFixed(3) : '—'}</td>
                                        <td><span class="badge ${this.getStatusBadge(c.status)}">${c.status}</span></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    },

    getStatusBadge(status) {
        const badges = {
            'passed': 'badge-success',
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

    attachEvents() {
        // Tab switching
        document.querySelectorAll('#results-tabs .tab').forEach(tab => {
            tab.addEventListener('click', () => this.switchTab(tab.dataset.tab));
        });

        // Chart type buttons
        document.querySelectorAll('.chart-type-btn').forEach(btn => {
            btn.addEventListener('click', () => this.switchChart(btn.dataset.chart));
        });

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