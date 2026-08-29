// LM Studio Auto Optimizer - UI Components

export const UI = {
    init() {
        this.setupTooltips();
        this.setupModals();
    },

    setupTooltips() {
        // Tooltips are handled via CSS
    },

    setupModals() {
        // Close modal on overlay click
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal-overlay')) {
                this.closeModal();
            }
        });

        // Close modal on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });
    },

    // Status Indicator
    updateStatusIndicator(status) {
        const indicator = document.getElementById('lm-status');
        if (!indicator) return;

        if (status.lm_studio?.connected) {
            indicator.className = 'status-badge status-online text-sm font-medium';
            indicator.innerHTML = '<span class="status-dot"></span>Connected';
        } else {
            indicator.className = 'status-badge status-offline text-sm font-medium';
            indicator.innerHTML = '<span class="status-dot"></span>Disconnected';
        }
    },

    // Dashboard Rendering
    renderDashboard(status, modelsResponse) {
        const models = modelsResponse.models || [];
        const hardware = status.hardware || {};

        return `
            <div class="space-y-8">
                <!-- LM Studio Status & Hardware -->
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    ${this.renderStatusCard(status.lm_studio || {})}
                    ${this.renderHardwareCard(hardware)}
                </div>

                <!-- Model Selection & Optimization -->
                <div class="card">
                    <div class="card-header">
                        <h2 class="text-lg font-semibold text-gray-900">Optimize Model</h2>
                    </div>
                    <div class="card-body">
                        ${this.renderOptimizationPanel(models)}
                    </div>
                </div>

                <!-- Quick Stats -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    ${this.renderQuickStats(models)}
                </div>
            </div>
        `;
    },

    renderStatusCard(lmStudio) {
        const connected = lmStudio.connected;
        const url = lmStudio.url || 'Not configured';
        const loaded = lmStudio.loaded_model;

        return `
            <div class="card">
                <div class="card-header">
                    <h3 class="font-semibold text-gray-900 flex items-center">
                        <svg class="w-5 h-5 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"></path>
                        </svg>
                        LM Studio
                    </h3>
                </div>
                <div class="card-body">
                    <div class="flex items-center justify-between mb-3">
                        <span class="status-badge ${connected ? 'status-online' : 'status-offline'}">
                            <span class="status-dot"></span>${connected ? 'Connected' : 'Disconnected'}
                        </span>
                    </div>
                    <div class="text-sm text-gray-600 font-mono break-all mb-3">${url}</div>
                    ${loaded ? `
                        <div class="p-3 bg-blue-50 rounded-lg">
                            <div class="text-xs text-gray-500">Currently Loaded</div>
                            <div class="font-mono text-sm font-medium">${loaded.name}</div>
                        </div>
                    ` : ''}
                    <button id="refresh-models" class="btn btn-outline btn-sm w-full mt-3">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                        </svg>
                        Refresh
                    </button>
                </div>
            </div>
        `;
    },

    renderHardwareCard(hardware) {
        const gpu = hardware.gpus?.[0];
        return `
            <div class="card">
                <div class="card-header">
                    <h3 class="font-semibold text-gray-900 flex items-center">
                        <svg class="w-5 h-5 mr-2 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"></path>
                        </svg>
                        Hardware
                    </h3>
                </div>
                <div class="card-body space-y-3">
                    ${gpu ? `
                        <div class="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                            <div>
                                <div class="text-xs text-gray-500">GPU</div>
                                <div class="font-mono text-sm font-medium">${gpu.name}</div>
                            </div>
                            <div class="text-right">
                                <div class="text-xs text-gray-500">VRAM</div>
                                <div class="font-mono text-lg font-semibold text-green-600">${gpu.vram_gb.toFixed(1)} GB</div>
                            </div>
                        </div>
                    ` : `
                        <div class="p-3 bg-gray-50 rounded-lg text-center text-gray-500">
                            No GPU detected
                        </div>
                    `}
                    <div class="grid grid-cols-2 gap-3">
                        <div class="p-3 bg-gray-50 rounded-lg">
                            <div class="text-xs text-gray-500">System RAM</div>
                            <div class="font-mono text-lg font-semibold">${(hardware.total_ram_gb || 0).toFixed(1)} GB</div>
                        </div>
                        <div class="p-3 bg-gray-50 rounded-lg">
                            <div class="text-xs text-gray-500">CPU</div>
                            <div class="font-mono text-sm font-medium truncate">${hardware.cpu_name || 'Unknown'}</div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    },

    renderOptimizationPanel(models) {
        // Sort models by name for deterministic display
        const sorted = [...models].sort((a,b) => a.name.localeCompare(b.name));
        const modelOptions = sorted.map(m => {
            const params = m.parameter_count ? `${(m.parameter_count/1e9).toFixed(1)}B` : 'N/A';
            const quant = m.quantization || 'unknown';
            const arch = m.architecture || 'unknown';
            const ctx = m.context_limit ? `${m.context_limit}` : 'N/A';
            const moe = m.is_moe ? 'MoE' : 'dense';
            const size = m.size_bytes ? `${(m.size_bytes/1024**3).toFixed(1)}GB` : '';
            const label = `${m.name} — ${params} ${quant} ${arch} ${ctx} ctx ${moe} ${size}`.trim();
            return `<option value="${m.id}" data-arch="${arch}" data-params="${m.parameter_count||''}" data-quant="${quant}" data-ctx="${m.context_limit||''}" data-moe="${m.is_moe||''}" data-size="${m.size_bytes||''}">${label} (${m.id})</option>`;
        }).join('');

        return `
            <div class="space-y-6">
                <!-- Model Selection - explicit, no auto-select -->
                <div>
                    <label class="form-label flex items-center justify-between">
                        Model <span class="text-xs text-red-500">*required — select one</span>
                    </label>
                    <input type="text" id="model-search" placeholder="Search models..." class="form-input mb-2" oninput="UI.filterModels(this.value)">
                    <select id="model-select" class="form-input form-select" required size="6" style="height:auto; min-height: 140px;">
                        <option value="" disabled selected>— Choose a model —</option>
                        ${modelOptions}
                    </select>
                    <div id="model-meta" class="mt-3 p-3 bg-blue-50 rounded-lg border border-blue-100 hidden"></div>
                    <p class="text-xs text-gray-500 mt-1">Different quantizations/versions are separate targets. Presets are model-specific and will not be applied across variants.</p>
                </div>

                <!-- Profile Selection - explicit with descriptions -->
                <div>
                    <label class="form-label">Optimization Profile <span class="text-xs text-red-500">*required</span></label>
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                        ${this.renderProfileOption('speed', 'Speed', 'Maximize generation & prompt speed, TTFT. Quality 0.95', '🚀')}
                        ${this.renderProfileOption('balanced', 'Balanced', 'Best compromise: speed, context, memory, correctness', '⚖️', true)}
                        ${this.renderProfileOption('context', 'Context', 'Maximize stable context (25% weight)', '📏')}
                        ${this.renderProfileOption('quality', 'Quality', 'Prioritize correctness (0.99) & stability', '✨')}
                        ${this.renderProfileOption('custom', 'Custom', 'Edit weights manually (must total 100%)', '🎛️')}
                    </div>
                    <div id="custom-weights" class="hidden mt-4 p-4 bg-gray-50 rounded-lg border">
                        <p class="text-sm font-medium mb-2">Custom Weights (must total 100%)</p>
                        <div class="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                            <label>Generation speed <input type="number" id="w-gen" value="30" min="0" max="100" class="form-input w-full">%</label>
                            <label>Prompt speed <input type="number" id="w-prompt" value="10" min="0" max="100" class="form-input w-full">%</label>
                            <label>Context <input type="number" id="w-ctx" value="20" min="0" max="100" class="form-input w-full">%</label>
                            <label>Memory <input type="number" id="w-mem" value="10" min="0" max="100" class="form-input w-full">%</label>
                            <label>Correctness <input type="number" id="w-qual" value="20" min="0" max="100" class="form-input w-full">%</label>
                            <label>Stability <input type="number" id="w-stab" value="10" min="0" max="100" class="form-input w-full">%</label>
                        </div>
                        <div id="weights-total" class="text-xs mt-2 text-gray-500">Total: 100% ✓</div>
                    </div>
                </div>

                <!-- Quality Threshold -->
                <div>
                    <label class="form-label flex items-center justify-between">
                        Quality Threshold
                        <span class="text-sm text-gray-500" id="quality-threshold-label">0.97</span>
                    </label>
                    <input type="range" id="quality-threshold" min="0.90" max="1.00" step="0.01" value="0.97" class="w-full">
                    <p class="text-xs text-gray-500 mt-1">Configs below this heuristic score are rejected. See docs/OPTIMIZATION_METHOD.md.</p>
                </div>

                <!-- Advanced Search Settings - collapsed by default, clear that normal users don't need it -->
                <div class="accordion-item border rounded-lg">
                    <div class="accordion-header flex items-center justify-between p-3 cursor-pointer bg-gray-50" onclick="this.parentElement.classList.toggle('open'); document.getElementById('advanced-content').classList.toggle('hidden')">
                        <span class="font-medium">Advanced Search Settings ▾</span>
                        <span class="text-xs text-gray-500">Normal users do not need to change these</span>
                    </div>
                    <div class="accordion-content hidden p-4" id="advanced-content">
                        <p class="text-xs text-gray-500 mb-4">Only supported parameters for your LM Studio version are shown. Unsupported options are hidden/disabled.</p>
                        ${this.renderAdvancedSettings()}
                    </div>
                </div>

                <!-- Pre-run Review Summary -->
                <div id="optimization-summary" class="p-4 bg-amber-50 rounded-lg border border-amber-200">
                    <p class="text-amber-800 text-sm">Select a model and profile to see search space review. You will be asked to confirm before running.</p>
                </div>

                <!-- Start Button - disabled until model selected -->
                <button id="start-optimization" class="btn btn-success w-full btn-lg opacity-50 cursor-not-allowed" disabled>
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                    </svg>
                    Select a model to start
                </button>
                <p class="text-xs text-center text-gray-400">Optimization will benchmark many configurations — you will see a review before it starts.</p>
            </div>
        `;
    },

    filterModels(query) {
        const sel = document.getElementById('model-select');
        if (!sel) return;
        const q = query.toLowerCase();
        for (const opt of sel.options) {
            if (!opt.value) continue;
            const txt = opt.textContent.toLowerCase();
            opt.hidden = q && !txt.includes(q);
        }
    },

    updateModelMeta(modelId, models) {
        const meta = document.getElementById('model-meta');
        const btn = document.getElementById('start-optimization');
        if (!modelId || !meta) {
            if (meta) meta.classList.add('hidden');
            if (btn) { btn.disabled = true; btn.classList.add('opacity-50','cursor-not-allowed'); btn.innerHTML = 'Select a model to start'; }
            return;
        }
        const m = models.find(x => x.id === modelId);
        if (!m) return;
        const params = m.parameter_count ? `${(m.parameter_count/1e9).toFixed(1)}B` : 'unknown';
        const quant = m.quantization || 'unknown';
        const arch = m.architecture || 'unknown';
        const ctx = m.context_limit || 'unknown';
        const moe = m.is_moe ? `MoE (${m.num_experts||'?' } experts)` : 'dense';
        const size = m.size_bytes ? `${(m.size_bytes/1024**3).toFixed(1)} GB` : '';
        meta.innerHTML = `
            <div class="grid grid-cols-2 md:grid-cols-3 gap-2 text-sm">
                <div><span class="text-gray-500">ID</span><br><span class="font-mono font-medium break-all">${m.id}</span></div>
                <div><span class="text-gray-500">Name</span><br><span class="font-medium">${m.name}</span></div>
                <div><span class="text-gray-500">Arch</span><br><span class="font-mono">${arch}</span></div>
                <div><span class="text-gray-500">Params</span><br><span class="font-mono">${params}</span></div>
                <div><span class="text-gray-500">Quant</span><br><span class="font-mono">${quant}</span></div>
                <div><span class="text-gray-500">Context</span><br><span class="font-mono">${ctx}</span></div>
                <div><span class="text-gray-500">Type</span><br><span class="font-mono">${moe}</span></div>
                <div><span class="text-gray-500">Size</span><br><span class="font-mono">${size}</span></div>
            </div>
        `;
        meta.classList.remove('hidden');
        if (btn) { btn.disabled = false; btn.classList.remove('opacity-50','cursor-not-allowed'); btn.innerHTML = 'Review & Start Optimization'; }
    },

    generatePreRunReview(modelId, profile, advanced, estimatedConfigs) {
        const est = estimatedConfigs || '~20–50';
        const isLarge = (typeof estimatedConfigs === 'number' && estimatedConfigs > 100);
        return `
            <div class="space-y-4">
                <h3 class="font-semibold text-gray-900">Optimization Review</h3>
                <div class="p-3 bg-blue-50 rounded-lg border border-blue-100 space-y-1 text-sm">
                    <div><span class="text-gray-500">Model</span><br><span class="font-mono font-medium">${modelId}</span></div>
                    <div><span class="text-gray-500">Profile</span><br><span class="badge badge-info">${profile}</span></div>
                    <div><span class="text-gray-500">Search space</span><br>
                        Context: ${advanced.min_context} → ${advanced.max_context}<br>
                        GPU: ${(advanced.min_gpu_ratio*100).toFixed(0)}% → ${(advanced.max_gpu_ratio*100).toFixed(0)}%<br>
                        Flash: ${advanced.test_flash_on && advanced.test_flash_off ? 'ON/OFF' : advanced.test_flash_on ? 'ON' : 'OFF'}<br>
                        KV Cache: ${advanced.test_kv_gpu && advanced.test_kv_cpu ? 'GPU/CPU' : advanced.test_kv_gpu ? 'GPU' : 'CPU'}<br>
                        Batch: ${advanced.auto_batch ? '64→512 (auto)' : 'custom'}<br>
                    </div>
                    <div><span class="text-gray-500">Estimated configurations</span><br><span class="font-mono font-medium">${est}</span></div>
                    <div><span class="text-gray-500">Benchmark repetitions</span> 3 &nbsp; <span class="text-gray-500">Validation</span> 5</div>
                </div>
                ${isLarge ? `<div class="p-3 bg-amber-100 border border-amber-300 rounded-lg text-amber-800 text-sm"><strong>Large optimization run</strong><br>Estimated tests: ${estimatedConfigs}<br>Estimated duration: unknown (hours). Continue?</div>` : ''}
                <p class="text-xs text-gray-500">Different quantizations/versions are separate optimization targets. A preset for one will not be silently applied to another.</p>
            </div>
        `;
    },

    renderProfileOption(value, name, description, icon, checked = false) {
        return `
            <label class="relative cursor-pointer">
                <input type="radio" name="profile" value="${value}" class="sr-only" ${checked ? 'checked' : ''}>
                <div class="p-4 border-2 rounded-lg ${checked ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'} transition-all">
                    <div class="text-2xl mb-1">${icon}</div>
                    <div class="font-semibold text-gray-900">${name}</div>
                    <div class="text-xs text-gray-500">${description}</div>
                </div>
            </label>
        `;
    },

    renderAdvancedSettings() {
        return `
            <div class="space-y-6">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label class="form-label">Min Context</label>
                        <input type="number" id="min-context" value="2048" class="form-input" min="512">
                    </div>
                    <div>
                        <label class="form-label">Max Context</label>
                        <input type="number" id="max-context" value="32768" class="form-input" min="512">
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label class="form-label">Min GPU Ratio</label>
                        <input type="number" id="min-gpu-ratio" value="0" step="0.05" min="0" max="1" class="form-input">
                    </div>
                    <div>
                        <label class="form-label">Max GPU Ratio</label>
                        <input type="number" id="max-gpu-ratio" value="1" step="0.05" min="0" max="1" class="form-input">
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="form-checkbox">
                        <input type="checkbox" id="test-flash-on" checked class="form-checkbox-input">
                        <label for="test-flash-on">Test Flash Attention ON</label>
                    </div>
                    <div class="form-checkbox">
                        <input type="checkbox" id="test-flash-off" checked class="form-checkbox-input">
                        <label for="test-flash-off">Test Flash Attention OFF</label>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="form-checkbox">
                        <input type="checkbox" id="test-kv-gpu" checked class="form-checkbox-input">
                        <label for="test-kv-gpu">Test KV Cache on GPU</label>
                    </div>
                    <div class="form-checkbox">
                        <input type="checkbox" id="test-kv-cpu" checked class="form-checkbox-input">
                        <label for="test-kv-cpu">Test KV Cache on CPU</label>
                    </div>
                </div>

                <div class="form-checkbox">
                    <input type="checkbox" id="auto-batch" checked class="form-checkbox-input">
                    <label for="auto-batch">Auto optimize eval batch size (64-1024)</label>
                </div>
            </div>
        `;
    },

    renderQuickStats(models) {
        const stats = [
            { label: 'Models Available', value: models.length, icon: '📦' },
            { label: 'Optimized', value: 0, icon: '⚡' }, // Would come from presets
            { label: 'Runs Today', value: 0, icon: '📊' },
        ];

        return stats.map(s => `
            <div class="card p-6 text-center">
                <div class="text-4xl mb-2">${s.icon}</div>
                <div class="text-3xl font-bold text-gray-900">${s.value}</div>
                <div class="text-sm text-gray-500">${s.label}</div>
            </div>
        `).join('');
    },

    // History Page
    renderHistory(runs) {
        if (!runs.length) {
            return `
                <div class="card">
                    <div class="card-body">
                        <div class="empty-state">
                            <svg class="empty-state-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                            </svg>
                            <h3 class="text-lg font-medium text-gray-900">No optimization runs yet</h3>
                            <p class="text-gray-500 mt-1">Start your first optimization from the Dashboard</p>
                        </div>
                    </div>
                </div>
            `;
        }

        return `
            <div class="card">
                <div class="card-header">
                    <h2 class="text-lg font-semibold text-gray-900">Optimization History</h2>
                </div>
                <div class="card-body p-0">
                    <div class="table-container">
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Model</th>
                                    <th>Profile</th>
                                    <th>Status</th>
                                    <th>Best Score</th>
                                    <th>Duration</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${runs.map(r => `
                                    <tr class="run-row hover:bg-gray-50 cursor-pointer" data-run-id="${r.id}">
                                        <td class="font-mono text-sm">${new Date(r.created_at).toLocaleString()}</td>
                                        <td>
                                            <div class="font-medium">${r.model.name}</div>
                                            <div class="text-xs text-gray-500 font-mono">${r.model.id}</div>
                                        </td>
                                        <td><span class="badge badge-info">${r.profile}</span></td>
                                        <td><span class="badge ${this.getStatusBadge(r.status)}">${r.status}</span></td>
                                        <td class="font-mono">${r.best_config_id ? '✓' : '—'}</td>
                                        <td class="font-mono">${r.duration_seconds ? r.duration_seconds.toFixed(1) + 's' : '—'}</td>
                                        <td>
                                            <button class="btn btn-outline btn-sm" onclick="event.stopPropagation(); window.location.href='/results/${r.id}'">View</button>
                                        </td>
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
            'completed': 'badge-success',
            'running': 'badge-warning',
            'paused': 'badge-info',
            'cancelled': 'badge-error',
            'failed': 'badge-error',
        };
        return badges[status] || 'badge-info';
    },

    // Settings Page
    renderSettings(settings) {
        return `
            <div class="max-w-3xl mx-auto space-y-8">
                <div>
                    <h2 class="text-2xl font-bold text-gray-900">Settings</h2>
                </div>

                <form id="settings-form" class="space-y-8">
                    <!-- LM Studio Connection -->
                    <div class="card">
                        <div class="card-header">
                            <h3 class="font-semibold text-gray-900">LM Studio Connection</h3>
                        </div>
                        <div class="card-body space-y-4">
                            <div>
                                <label class="form-label">LM Studio URL</label>
                                <div class="flex gap-2">
                                    <input type="url" id="lm-studio-url" value="${settings.lm_studio_url}" class="form-input flex-1">
                                    <button type="button" id="test-connection" class="btn btn-outline">Test Connection</button>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Web UI -->
                    <div class="card">
                        <div class="card-header">
                            <h3 class="font-semibold text-gray-900">Web UI</h3>
                        </div>
                        <div class="card-body space-y-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label class="form-label">Host</label>
                                <input type="text" id="web-host" name="web_host" value="${settings.web_host}" class="form-input">
                            </div>
                            <div>
                                <label class="form-label">Port</label>
                                <input type="number" id="web-port" name="web_port" value="${settings.web_port}" class="form-input">
                            </div>
                        </div>
                    </div>

                    <!-- Defaults -->
                    <div class="card">
                        <div class="card-header">
                            <h3 class="font-semibold text-gray-900">Default Optimization Settings</h3>
                        </div>
                        <div class="card-body space-y-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label class="form-label">Default Profile</label>
                                <select name="default_profile" class="form-input form-select">
                                    <option value="speed" ${settings.default_profile === 'speed' ? 'selected' : ''}>Speed</option>
                                    <option value="balanced" ${settings.default_profile === 'balanced' ? 'selected' : ''}>Balanced</option>
                                    <option value="context" ${settings.default_profile === 'context' ? 'selected' : ''}>Context</option>
                                    <option value="quality" ${settings.default_profile === 'quality' ? 'selected' : ''}>Quality</option>
                                </select>
                            </div>
                            <div>
                                <label class="form-label">Default Quality Threshold</label>
                                <input type="number" name="default_quality_threshold" value="${settings.default_quality_threshold}" step="0.01" min="0.9" max="1" class="form-input">
                            </div>
                            <div>
                                <label class="form-label">Benchmark Repetitions</label>
                                <input type="number" name="default_benchmark_repetitions" value="${settings.default_benchmark_repetitions}" min="1" max="10" class="form-input">
                            </div>
                            <div>
                                <label class="form-label">Validation Repetitions</label>
                                <input type="number" name="default_validation_repetitions" value="${settings.default_validation_repetitions}" min="1" max="10" class="form-input">
                            </div>
                        </div>
                    </div>

                    <!-- Behavior -->
                    <div class="card">
                        <div class="card-header">
                            <h3 class="font-semibold text-gray-900">Behavior</h3>
                        </div>
                        <div class="card-body space-y-4">
                            <div class="form-checkbox">
                                <input type="checkbox" name="auto_unload_after_tests" id="auto-unload" ${settings.auto_unload_after_tests ? 'checked' : ''} class="form-checkbox-input">
                                <label for="auto-unload">Auto unload model after each test</label>
                            </div>
                            <div>
                                <label class="form-label">Timeout (seconds)</label>
                                <input type="number" name="timeout_seconds" value="${settings.timeout_seconds}" min="60" max="3600" class="form-input">
                            </div>
                        </div>
                    </div>

                    <!-- Save Button -->
                    <div class="flex justify-end">
                        <button type="submit" class="btn btn-success">Save Settings</button>
                    </div>
                </form>
            </div>
        `;
    },

    // Optimization Summary
    generateOptimizationSummary(modelId, profile, advancedEnabled) {
        // This would ideally fetch the search space from the API
        // For now, show a placeholder
        return `
            <div class="space-y-3">
                <div class="flex items-center justify-between">
                    <span class="font-medium">Model:</span>
                    <span class="font-mono text-sm">${modelId}</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="font-medium">Profile:</span>
                    <span class="badge badge-info">${profile}</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="font-medium">Quality Threshold:</span>
                    <span class="font-mono">0.97</span>
                </div>
                <div class="pt-2 border-t border-gray-200">
                    <p class="text-sm text-gray-500">Click "Start Optimization" to begin. Estimated configurations: ~20-50 depending on settings.</p>
                </div>
            </div>
        `;
    },

    // Error & Toast
    renderError(message) {
        return `
            <div class="card">
                <div class="card-body text-center py-12">
                    <svg class="mx-auto h-12 w-12 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
                    </svg>
                    <h3 class="mt-2 text-lg font-medium text-gray-900">Error</h3>
                    <p class="mt-1 text-gray-500">${message}</p>
                </div>
            </div>
        `;
    },

    showToast(message, type = 'info') {
        const container = this.getToastContainer();
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="flex items-center gap-2 p-4 bg-white rounded-lg shadow-lg border-l-4 ${this.getToastBorder(type)}">
                <svg class="w-5 h-5 ${this.getToastIconColor(type)}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    ${this.getToastIcon(type)}
                </svg>
                <span class="text-sm font-medium text-gray-900">${message}</span>
            </div>
        `;

        container.appendChild(toast);
        setTimeout(() => toast.remove(), 5000);
    },

    getToastContainer() {
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'fixed bottom-4 right-4 z-50 space-y-2';
            document.body.appendChild(container);
        }
        return container;
    },

    getToastBorder(type) {
        return {
            success: 'border-green-500',
            error: 'border-red-500',
            warning: 'border-yellow-500',
            info: 'border-blue-500',
        }[type] || 'border-blue-500';
    },

    getToastIconColor(type) {
        return {
            success: 'text-green-500',
            error: 'text-red-500',
            warning: 'text-yellow-500',
            info: 'text-blue-500',
        }[type] || 'text-blue-500';
    },

    getToastIcon(type) {
        return {
            success: '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>',
            error: '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>',
            warning: '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>',
            info: '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>',
        }[type] || '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>';
    },

    showLoading(message = 'Loading...') {
        const overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        overlay.innerHTML = `
            <div class="bg-white p-8 rounded-lg shadow-xl text-center">
                <div class="spinner mx-auto mb-4"></div>
                <p class="text-gray-700">${message}</p>
            </div>
        `;
        document.body.appendChild(overlay);
    },

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) overlay.remove();
    },

    closeModal() {
        const modal = document.querySelector('.modal-overlay');
        if (modal) modal.remove();
    },

    openModal(content, title = '') {
        this.closeModal();
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3 class="text-lg font-semibold">${title}</h3>
                    <button onclick="UI.closeModal()" class="text-gray-400 hover:text-gray-600 text-2xl">&times;</button>
                </div>
                <div class="modal-body">${content}</div>
            </div>
        `;
        document.body.appendChild(modal);
    },
};