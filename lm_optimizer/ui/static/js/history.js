// LM Studio Auto Optimizer - History Page

import { API } from './api.js?v=13';
import { UI } from './ui.js?v=13';

const HistoryPage = {
    state: {
        runs: [],
        filteredRuns: [],
        currentFilter: { model: '', profile: '', status: '' },
    },

    async init() {
        UI.init();
        await this.loadRuns();
        await this.loadDuels();
        this.render();
        this.attachEvents();
        // Deep link from the Sandbox page to its history section.
        if (window.location.hash === '#sandbox-history') {
            const el = document.getElementById('sandbox-history');
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    },

    async loadDuels() {
        try {
            const response = await API.getDuels(100);
            this.state.duels = response.duels || [];
        } catch (error) {
            console.error('Failed to load duels:', error);
            this.state.duels = [];
            this.state.duelsError = (error && error.message) || String(error);
        }
    },

    async loadRuns() {
        try {
            const response = await API.getRuns(100);
            this.state.runs = response.runs || [];
            this.state.filteredRuns = this.state.runs;
            this.state.loadError = null;
        } catch (error) {
            console.error('Failed to load runs:', error);
            this.state.loadError = (error && error.message) || String(error);
            this.state.runs = [];
            this.state.filteredRuns = [];
        }
    },

    render() {
        const main = document.getElementById('main-content');
        const err = this.state.loadError
            ? `<div class="card border-red-300"><div class="card-body text-center py-8">`
              + `<h3 class="text-lg font-medium text-red-700">History failed to load.</h3>`
              + `<p class="text-sm text-gray-500 mt-1">${String(this.state.loadError).slice(0, 300)}</p>`
              + `<button class="btn btn-outline mt-4" onclick="window.location.reload()">Retry</button>`
              + `</div></div>`
            : '';
        main.innerHTML = `
            <div class="space-y-8">
                ${err}
                <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                    <div>
                        <h1 class="text-3xl font-bold text-gray-900">Optimization History</h1>
                        <p class="text-gray-500 mt-1">View and manage all optimization runs</p>
                    </div>
                    <div class="flex gap-2">
                        <button id="resume-all-pending" class="btn btn-success" title="Resume all paused runs from checkpoints">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path>
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                            </svg>
                            Resume All Pending
                        </button>
                        <button id="refresh-history" class="btn btn-outline">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                            </svg>
                            Refresh
                        </button>
                        <button id="reconcile-runs" class="btn btn-outline" title="Mark dead 'running' runs interrupted (keeps history + checkpoints)">
                            Reconcile Stale
                        </button>
                    </div>
                </div>

                <!-- Filters -->
                <div class="card">
                    <div class="card-body">
                        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                            <div>
                                <label class="form-label">Filter by Model</label>
                                <select id="filter-model" class="form-input form-select">
                                    <option value="">All Models</option>
                                    ${this.renderModelOptions()}
                                </select>
                            </div>
                            <div>
                                <label class="form-label">Filter by Profile</label>
                                <select id="filter-profile" class="form-input form-select">
                                    <option value="">All Profiles</option>
                                    <option value="speed">Speed</option>
                                    <option value="balanced">Balanced</option>
                                    <option value="context">Context</option>
                                    <option value="quality">Quality</option>
                                </select>
                            </div>
                            <div>
                                <label class="form-label">Filter by Status</label>
                                <select id="filter-status" class="form-input form-select">
                                    <option value="">All Statuses</option>
                                    <option value="success">Success (validated winner)</option>
                                    <option value="completed_no_validated_winner">Completed, no validated winner</option>
                                    <option value="partial_success">Partial success</option>
                                    <option value="running">Running</option>
                                    <option value="paused">Paused</option>
                                    <option value="interrupted">Interrupted</option>
                                    <option value="failed">Failed</option>
                                    <option value="cancelled">Cancelled</option>
                                </select>
                            </div>
                            <div class="flex items-end">
                                <button id="clear-filters" class="btn btn-outline w-full">Clear Filters</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Runs Table -->
                <div class="card" id="runs-table-container">
                    ${this.renderRunsTable()}
                </div>

                <!-- Sandbox Duels -->
                <div class="flex items-center justify-between mt-8 mb-2" id="sandbox-history">
                    <h2 class="text-xl font-bold text-gray-900">Sandbox Duels</h2>
                    <button id="delete-duels" class="btn btn-outline btn-sm" disabled>Delete selected</button>
                </div>
                <div class="card" id="duels-table-container">
                    ${this.renderDuelsTable()}
                </div>
            </div>
        `;

        this.attachEvents();
    },

    renderDuelsTable() {
        const duels = this.state.duels || [];
        if (this.state.duelsError && !duels.length) {
            return `<div class="card-body"><p class="text-sm text-red-600">Duels failed to load: ${this.state.duelsError}</p></div>`;
        }
        if (!duels.length) {
            return `<div class="card-body"><p class="text-sm text-gray-500">No duels yet — run one from the Sandbox page.</p></div>`;
        }
        return `
            <div class="card-body p-0">
                <div class="table-container">
                    <table class="table">
                        <thead><tr>
                            <th></th><th>Date</th><th>Kind</th><th>Model A vs B</th>
                            <th>Status</th><th>Faster</th><th>Actions</th>
                        </tr></thead>
                        <tbody>
                            ${duels.map(d => `
                                <tr class="duel-row hover:bg-gray-50" data-duel-id="${d.id}">
                                    <td><input type="checkbox" class="duel-checkbox" data-id="${d.id}"></td>
                                    <td class="font-mono text-sm">${new Date(d.created_at).toLocaleString()}</td>
                                    <td><span class="badge badge-info">${d.kind}</span></td>
                                    <td><div class="font-medium text-sm font-mono">${d.model_a}</div>
                                        <div class="text-xs text-gray-500 font-mono">vs ${d.model_b}</div></td>
                                    <td><span class="badge ${UI.getStatusBadge(d.status)}">${d.status}</span>${d.stale ? ' <span class="badge badge-warning" title="Still running with no completion">stale</span>' : ''}</td>
                                    <td class="font-mono">${d.faster || '—'}</td>
                                    <td><button class="btn btn-outline btn-sm view-duel" data-id="${d.id}">View</button></td>
                                </tr>
                                <tr class="duel-detail-row hidden" data-duel-detail="${d.id}">
                                    <td colspan="7"><div class="p-3 duel-detail-body" data-duel-body="${d.id}"></div></td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    },

    async toggleDuelDetail(id) {
        const row = document.querySelector(`[data-duel-detail="${id}"]`);
        const body = document.querySelector(`[data-duel-body="${id}"]`);
        if (!row || !body) return;
        if (!body.dataset.loaded) {
            body.innerHTML = '<p class="text-sm text-gray-500">Loading...</p>';
            try {
                const d = await API.getDuel(id);
                body.innerHTML = this.renderDuelDetail(d);
            } catch (e) {
                body.innerHTML = `<p class="text-sm text-red-600">Failed: ${String((e && e.message) || e).slice(0, 200)}</p>`;
                row.classList.remove('hidden');
                return;
            }
            body.dataset.loaded = '1';
        }
        row.classList.toggle('hidden');
    },

    renderDuelDetail(d) {
        let result = null;
        try {
            result = typeof d.result_json === 'string' ? JSON.parse(d.result_json) : d.result_json;
        } catch (e) {
            result = null;
        }
        const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (c) => (
            { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
        let inner = `<p class="text-sm"><span class="font-medium">Prompt:</span> ${esc((d.prompt || '').slice(0, 2000))}</p>`;
        if (!result) {
            return inner + `<p class="text-sm text-gray-500 mt-1">${d.status === 'running' ? 'Still running...' : 'No result stored.'}</p>`;
        }
        const sideBlock = (label, s) => {
            if (!s) return '';
            if (s.file_path) {
                const parts = String(s.file_path).split('/');
                const url = `/sandbox/files/${parts[0]}/${parts[1]}/index.html`;
                return `<div><h4 class="font-bold">${label}</h4>`
                    + `<p class="text-xs font-mono text-gray-500">results/sandbox/${esc(s.file_path)}</p>`
                    + `<p class="mt-1"><a class="underline text-blue-600" target="_blank" rel="noopener" href="${url}">Open index.html</a></p></div>`;
            }
            return `<div><h4 class="font-bold">${label} <span class="text-xs font-normal text-gray-500">${s.tok_s ? s.tok_s.toFixed(1) + ' tok/s' : ''}</span></h4>`
                + `<pre class="font-mono text-xs bg-gray-50 p-2 rounded overflow-auto max-h-64 mt-1">${esc((s.text || s.error || '').slice(0, 6000))}</pre></div>`;
        };
        const sa = result.side_a, sb = result.side_b;
        inner += `<div class="grid md:grid-cols-2 gap-4 mt-2">${sideBlock('A', sa)}${sideBlock('B', sb)}</div>`;
        return inner;
    },

    renderModelOptions() {
        const models = [...new Set(this.state.runs.map(r => r.model.id))];
        return models.map(m => {
            const run = this.state.runs.find(r => r.model.id === m);
            return `<option value="${m}">${run?.model.name || m}</option>`;
        }).join('');
    },

    renderRunsTable() {
        if (!this.state.filteredRuns.length) {
            return `
                <div class="card-body">
                    <div class="empty-state">
                        <svg class="empty-state-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                        </svg>
                        <h3 class="text-lg font-medium text-gray-900">No optimization runs found</h3>
                        <p class="text-gray-500 mt-1">Start your first optimization from the Dashboard</p>
                    </div>
                </div>
            `;
        }

        return `
            <div class="card-body border-b flex items-center gap-2 flex-wrap">
                <span id="bulk-count" class="text-sm text-gray-500">0 selected</span>
                <button id="bulk-compare" class="btn btn-outline btn-sm" disabled>Compare selected</button>
                <button id="bulk-rerun" class="btn btn-outline btn-sm" disabled>Re-run selected</button>
                <button id="bulk-delete" class="btn btn-danger btn-sm" disabled>Delete selected</button>
                <button id="bulk-clear" class="btn btn-outline btn-sm">Clear</button>
            </div>
            <div class="card-body p-0">
                <div class="table-container">
                    <table class="table history-table">
                        <thead>
                            <tr>
                                <th class="w-10"><input type="checkbox" id="select-all-runs" title="Select all visible runs"></th>
                                <th>Date</th>
                                <th>Model</th>
                                <th>Profile</th>
                                <th>Status</th>
                                <th>Best Gen tok/s</th>
                                <th>Quality</th>
                                <th>Duration</th>
                                <th>Configurations</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${this.state.filteredRuns.map(r => `
                                <tr class="run-row hover:bg-gray-50 cursor-pointer" data-run-id="${r.id}">
                                    <td class="text-center"><input type="checkbox" class="run-select" value="${r.id}"></td>
                                    <td class="font-mono text-sm">${new Date(r.created_at).toLocaleString()}</td>
                                    <td class="model-cell">
                                        <div class="font-medium">${r.model.name}</div>
                                        <div class="text-xs text-gray-500 font-mono">${r.model.id}</div>
                                    </td>
                                    <td><span class="badge badge-info">${r.profile}</span></td>
                                    <td><span class="badge ${UI.getStatusBadge(r.status)}">${r.status}</span>${r.stale ? ' <span class="badge badge-warning" title="No fresh checkpoint: process is gone">stale</span> <button class="btn btn-outline btn-sm abandon-run" data-run-id="${r.id}" title="Mark interrupted (keeps history)">Abandon</button>' : ''}</td>
                                    <td class="font-mono">${this.getBestGenSpeed(r)}</td>
                                    <td class="font-mono">${this.getBestQuality(r)}</td>
                                    <td class="font-mono">${r.duration_seconds ? r.duration_seconds.toFixed(1) + 's' : '—'}</td>
                                    <td class="font-mono">${r.config_count ?? r.configurations?.length ?? 0}</td>
                                    <td>
                                        <div class="flex gap-1">
                                            <button class="btn btn-outline btn-sm view-run" data-run-id="${r.id}">View</button>
                                            <button class="btn btn-outline btn-sm delete-run" data-run-id="${r.id}" title="Delete run">🗑</button>
                                        </div>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
    },

    getBestGenSpeed(run) {
        // Server-computed winner metrics (list rows carry no full configs).
        if (run.best_gen_tok_s !== undefined && run.best_gen_tok_s !== null) {
            return Number(run.best_gen_tok_s).toFixed(1);
        }
        if (run.configurations && run.best_config_id) {
            const best = run.configurations.find(c => c.id === run.best_config_id);
            if (best?.avg_generation_tok_s) return best.avg_generation_tok_s.toFixed(1);
        }
        return '—';
    },

    getBestQuality(run) {
        if (run.best_quality !== undefined && run.best_quality !== null) {
            return Number(run.best_quality).toFixed(3);
        }
        if (run.configurations && run.best_config_id) {
            const best = run.configurations.find(c => c.id === run.best_config_id);
            if (best?.quality?.overall) return best.quality.overall.toFixed(3);
        }
        return '—';
    },

    attachEvents() {
        // Resume All Pending
        document.getElementById('resume-all-pending')?.addEventListener('click', () => this.resumeAllPending());

        // Reconcile stale running runs (preview → confirm → apply).
        document.getElementById('reconcile-runs')?.addEventListener('click', () => this.reconcileStaleRuns());

        // Refresh
        document.getElementById('refresh-history')?.addEventListener('click', () => this.loadRuns());

        // Filters
        ['model', 'profile', 'status'].forEach(key => {
            const el = document.getElementById(`filter-${key}`);
            if (el) {
                el.addEventListener('change', () => this.applyFilters());
            }
        });

        // Clear filters
        document.getElementById('clear-filters')?.addEventListener('click', () => {
            ['model', 'profile', 'status'].forEach(key => {
                const el = document.getElementById(`filter-${key}`);
                if (el) el.value = '';
            });
            this.applyFilters();
        });

        // Bulk selection (delete / re-run / compare)
        const boxes = () => Array.from(document.querySelectorAll('.run-select'));
        const selectedIds = () => boxes().filter(b => b.checked).map(b => b.value);
        const updateBulk = () => {
            const n = selectedIds().length;
            const count = document.getElementById('bulk-count');
            if (count) count.textContent = `${n} selected`;
            const del = document.getElementById('bulk-delete');
            const rerun = document.getElementById('bulk-rerun');
            const cmp = document.getElementById('bulk-compare');
            if (del) del.disabled = n === 0;
            if (rerun) rerun.disabled = n === 0;
            if (cmp) cmp.disabled = n < 2;
        };
        boxes().forEach(b => {
            b.addEventListener('click', e => e.stopPropagation());
            b.addEventListener('change', updateBulk);
        });
        document.getElementById('select-all-runs')?.addEventListener('change', (e) => {
            boxes().forEach(b => { b.checked = e.target.checked; });
            updateBulk();
        });
        document.getElementById('bulk-clear')?.addEventListener('click', () => {
            boxes().forEach(b => { b.checked = false; });
            const all = document.getElementById('select-all-runs');
            if (all) all.checked = false;
            updateBulk();
        });
        document.getElementById('bulk-delete')?.addEventListener('click', async () => {
            const ids = selectedIds();
            if (!ids.length) return;
            if (!window.confirm(`Delete ${ids.length} selected run(s) and their configurations?`)) return;
            let ok = 0, failed = 0;
            for (const id of ids) {
                try { await API.deleteRun(id); ok++; }
                catch (err) { failed++; }
            }
            UI.showToast(`Deleted ${ok} run(s)${failed ? `, ${failed} failed (live runs are refused)` : ''}`, failed ? 'warning' : 'success');
            await this.refresh();
        });
        document.getElementById('bulk-rerun')?.addEventListener('click', async () => {
            const ids = selectedIds();
            if (!ids.length) return;
            const run = (this.state.runs || []).find(r => r.id === ids[0]);
            if (!run) { UI.showToast('Selected run not found', 'error'); return; }
            try {
                await API.startOptimization({
                    model_id: run.model.id,
                    profile: run.profile,
                    advanced_settings: {},
                });
                UI.showToast(`Re-run started: ${run.model.id}${ids.length > 1 ? ' (only the first selected model was started)' : ''}`, 'success');
            } catch (err) {
                UI.showToast((err && err.message) || 'Re-run failed', 'error');
            }
        });
        document.getElementById('bulk-compare')?.addEventListener('click', async () => {
            const ids = selectedIds();
            if (ids.length < 2) return;
            UI.showLoading('Judging selected runs...');
            try {
                const res = await API.judge({
                    run_ids: ids,
                    backend: 'heuristic',
                    priority: 'balanced',
                    wait: true,
                });
                UI.hideLoading();
                const verdict = (res && res.verdict) || {};
                const text = JSON.stringify(verdict, null, 2)
                    .replace(/&/g, '&amp;').replace(/</g, '&lt;');
                UI.openModal(
                    '<p class="text-sm text-gray-600 mb-2">Heuristic comparison of '
                    + ids.length + ' selected runs.</p>'
                    + '<pre class="text-xs whitespace-pre-wrap bg-gray-50 p-3 rounded max-h-[60vh] overflow-auto">'
                    + text + '</pre>',
                    'Compare selected runs'
                );
            } catch (err) {
                UI.hideLoading();
                UI.showToast((err && err.message) || 'Judge failed', 'error');
            }
        });

        // View run buttons
        document.querySelectorAll('.view-run').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const runId = btn.dataset.runId;
                window.location.href = `/results/${runId}`;
            });
        });

        // Row click
        document.querySelectorAll('.run-row').forEach(row => {
            row.addEventListener('click', () => {
                window.location.href = `/results/${row.dataset.runId}`;
            });
        });

        // Abandon stale runs (keeps them as interrupted history)
        document.querySelectorAll('.abandon-run').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                if (!window.confirm('Mark this dead run as interrupted? (stays in history)')) return;
                try {
                    await API.abandonRun(btn.dataset.runId);
                    UI.showToast('Run marked interrupted', 'success');
                    await this.refresh();
                } catch (err) {
                    UI.showToast((err && err.message) || 'Abandon failed', 'error');
                }
            });
        });

        // Delete run buttons (server refuses live runs with 409)
        document.querySelectorAll('.delete-run').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                if (!window.confirm('Delete this run and its configurations?')) return;
                try {
                    await API.deleteRun(btn.dataset.runId);
                    UI.showToast('Run deleted', 'success');
                    await this.refresh();
                } catch (err) {
                    UI.showToast((err && err.message) || 'Delete failed', 'error');
                }
            });
        });

        // Duel expand
        document.querySelectorAll('.view-duel').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleDuelDetail(btn.dataset.id);
            });
        });
        document.querySelectorAll('.duel-row').forEach(row => {
            row.addEventListener('click', (e) => {
                if (e.target.closest('button') || e.target.closest('input')) return;
                this.toggleDuelDetail(row.dataset.duelId);
            });
        });

        // Duel checkboxes + bulk delete
        const delBtn = document.getElementById('delete-duels');
        const syncDel = () => {
            const n = document.querySelectorAll('.duel-checkbox:checked').length;
            if (delBtn) {
                delBtn.disabled = n === 0;
                delBtn.textContent = n ? `Delete selected (${n})` : 'Delete selected';
            }
        };
        document.querySelectorAll('.duel-checkbox').forEach(cb => {
            cb.addEventListener('click', (e) => e.stopPropagation());
            cb.addEventListener('change', syncDel);
        });
        if (delBtn) {
            delBtn.addEventListener('click', async () => {
                const ids = [...document.querySelectorAll('.duel-checkbox:checked')]
                    .map(cb => cb.dataset.id);
                if (!ids.length) return;
                if (!window.confirm(`Delete ${ids.length} duel(s) including generated files?`)) return;
                for (const id of ids) {
                    try {
                        await API.deleteDuel(id);
                    } catch (err) {
                        UI.showToast(`Failed ${id}: ${(err && err.message) || err}`, 'error');
                    }
                }
                await this.loadDuels();
                document.getElementById('duels-table-container').innerHTML = this.renderDuelsTable();
                this.attachEvents();
            });
        }
    },

    applyFilters() {
        const model = document.getElementById('filter-model')?.value || '';
        const profile = document.getElementById('filter-profile')?.value || '';
        const status = document.getElementById('filter-status')?.value || '';

        this.state.filteredRuns = this.state.runs.filter(r => {
            if (model && r.model.id !== model) return false;
            if (profile && r.profile !== profile) return false;
            if (status && r.status !== status) return false;
            return true;
        });

        // Re-render table
        const container = document.getElementById('runs-table-container');
        container.innerHTML = this.renderRunsTable();

        // Re-attach view button events
        document.querySelectorAll('.view-run').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                window.location.href = `/results/${btn.dataset.runId}`;
            });
        });

        // Re-attach row click
        document.querySelectorAll('.run-row').forEach(row => {
            row.addEventListener('click', () => {
                window.location.href = `/results/${row.dataset.runId}`;
            });
        });
    },

    async refresh() {
        await this.loadRuns();
        await this.loadDuels();
        this.state.filteredRuns = this.state.runs;
        this.render(); // full re-render: fresh nodes, single attachEvents
    },

    async reconcileStaleRuns() {
        const btn = document.getElementById('reconcile-runs');
        if (btn) btn.disabled = true;
        try {
            const preview = await API.reconcileRuns(false);
            const changed = (preview && preview.changed) || 0;
            if (!changed) {
                UI.showToast('No stale running runs found', 'info');
                return;
            }
            if (!window.confirm(`Mark ${changed} dead 'running' run(s) as interrupted? History and checkpoints are preserved.`)) {
                return;
            }
            const applied = await API.reconcileRuns(true);
            UI.showToast(`Reconciled ${(applied && applied.changed) || 0} stale run(s)`, 'success');
            await this.refresh();
        } catch (err) {
            UI.showToast((err && err.message) || 'Reconcile failed', 'error');
        } finally {
            if (btn) btn.disabled = false;
        }
    },

    async resumeAllPending() {
        const btn = document.getElementById('resume-all-pending');
        if (btn) btn.disabled = true;
        UI.showLoading('Resuming all unfinished runs...');
        try {
            // Finishable states: paused, partial_success, interrupted.
            // Live (running/resumed) and explicitly cancelled runs are left alone.
            const resumable = ['paused', 'partial_success', 'interrupted'];
            const pendingRuns = this.state.runs.filter(r => resumable.includes(r.status));
            if (!pendingRuns.length) {
                UI.hideLoading();
                UI.showToast('No unfinished runs to resume', 'info');
                if (btn) btn.disabled = false;
                return;
            }
            let successCount = 0;
            let errorCount = 0;
            for (const run of pendingRuns) {
                try {
                    await API.resumeRunFromCheckpoint(run.id);
                    successCount++;
                } catch (error) {
                    errorCount++;
                    console.error(`Failed to resume ${run.id}:`, error);
                }
            }
            UI.hideLoading();
            if (successCount > 0) {
                UI.showToast(`Resumed ${successCount} run(s)${errorCount ? `, ${errorCount} failed` : ''}`, 'success');
            } else {
                UI.showToast(`All ${errorCount} resume attempts failed`, 'error');
            }
            await this.refresh();
        } catch (error) {
            UI.hideLoading();
            UI.showToast(error.message || 'Resume all failed', 'error');
            if (btn) btn.disabled = false;
        }
    },
};

// readyState-safe boot: deferred modules may evaluate after DOMContentLoaded.
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => HistoryPage.init());
} else {
    HistoryPage.init();
}