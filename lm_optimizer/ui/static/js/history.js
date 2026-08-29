// LM Studio Auto Optimizer - History Page

import { API } from './api.js';
import { UI } from './ui.js';

const HistoryPage = {
    state: {
        runs: [],
        filteredRuns: [],
        currentFilter: { model: '', profile: '', status: '' },
    },

    async init() {
        UI.init();
        await this.loadRuns();
        this.render();
        this.attachEvents();
    },

    async loadRuns() {
        try {
            const response = await API.getRuns(100);
            this.state.runs = response.runs || [];
            this.state.filteredRuns = this.state.runs;
        } catch (error) {
            console.error('Failed to load runs:', error);
            UI.showToast('Failed to load history', 'error');
        }
    },

    render() {
        const main = document.getElementById('main-content');
        main.innerHTML = `
            <div class="space-y-8">
                <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                    <div>
                        <h1 class="text-3xl font-bold text-gray-900">Optimization History</h1>
                        <p class="text-gray-500 mt-1">View and manage all optimization runs</p>
                    </div>
                    <div class="flex gap-2">
                        <button id="refresh-history" class="btn btn-outline">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                            </svg>
                            Refresh
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
                                    <option value="completed">Completed</option>
                                    <option value="running">Running</option>
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
            </div>
        `;

        this.attachEvents();
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
            <div class="card-body p-0">
                <div class="table-container">
                    <table class="table">
                        <thead>
                            <tr>
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
                                    <td class="font-mono text-sm">${new Date(r.created_at).toLocaleString()}</td>
                                    <td>
                                        <div class="font-medium">${r.model.name}</div>
                                        <div class="text-xs text-gray-500 font-mono">${r.model.id}</div>
                                    </td>
                                    <td><span class="badge badge-info">${r.profile}</span></td>
                                    <td><span class="badge ${UI.getStatusBadge(r.status)}">${r.status}</span></td>
                                    <td class="font-mono">${this.getBestGenSpeed(r)}</td>
                                    <td class="font-mono">${this.getBestQuality(r)}</td>
                                    <td class="font-mono">${r.duration_seconds ? r.duration_seconds.toFixed(1) + 's' : '—'}</td>
                                    <td class="font-mono">${r.configurations?.length || 0}</td>
                                    <td>
                                        <button class="btn btn-outline btn-sm view-run" data-run-id="${r.id}">View</button>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
    },

    getBestGenSpeed(run) {
        if (!run.best_config_id || !run.configurations) return '—';
        const best = run.configurations.find(c => c.id === run.best_config_id);
        return best?.avg_generation_tok_s ? best.avg_generation_tok_s.toFixed(1) : '—';
    },

    getBestQuality(run) {
        if (!run.best_config_id || !run.configurations) return '—';
        const best = run.configurations.find(c => c.id === run.best_config_id);
        return best?.quality?.overall ? best.quality.overall.toFixed(3) : '—';
    },

    attachEvents() {
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
        this.state.filteredRuns = this.state.runs;
        const container = document.getElementById('runs-table-container');
        container.innerHTML = this.renderRunsTable();
        this.attachEvents();
    },
};

document.addEventListener('DOMContentLoaded', () => HistoryPage.init());