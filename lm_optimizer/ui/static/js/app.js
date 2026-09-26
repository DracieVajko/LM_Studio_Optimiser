// LM Studio Auto Optimizer - Main Application

import { API } from './api.js?v=4';
import { UI } from './ui.js?v=4';
import { Charts } from './charts.js?v=4';
import { WebSocketManager } from './websocket.js?v=4';

// Application State
const App = {
    state: {
        currentView: 'dashboard',
        currentRunId: null,
        wsManager: null,
    },

    async init() {
        try {
            // Initialize UI
            UI.init();

            // Initialize WebSocket (optional: enhances live updates only)
            try {
                this.state.wsManager = new WebSocketManager();
            } catch (e) {
                console.warn('Live updates unavailable — refresh to update status.', e);
                this.state.wsManager = null;
            }

            // Load initial data
            await this.loadDashboard();

            // Set up navigation
            this.setupNavigation();

            // Periodic status updates
            setInterval(() => this.updateStatus(), 30000);
        } catch (error) {
            console.error('Dashboard init failed:', error);
            const main = document.getElementById('main-content');
            if (main) main.innerHTML = UI.renderError(
                'Dashboard failed to load: ' + ((error && error.message) || error),
                true
            );
        }
    },

    setupNavigation() {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const view = link.getAttribute('href').slice(1);
                this.navigateTo(view);
            });
        });
    },

    navigateTo(view) {
        this.state.currentView = view;
        document.querySelectorAll('.nav-link').forEach(l => {
            l.classList.toggle('active', l.getAttribute('href').slice(1) === view);
        });
        this.loadView(view);
    },

    async loadView(view) {
        const main = document.getElementById('main-content');
        main.innerHTML = '<div class="flex justify-center py-12"><div class="spinner"></div></div>';

        try {
            switch (view) {
                case 'dashboard':
                    await this.loadDashboard();
                    break;
                case 'history':
                    await this.loadHistory();
                    break;
                case 'settings':
                    await this.loadSettings();
                    break;
                default:
                    await this.loadDashboard();
            }
        } catch (error) {
            console.error('Failed to load view:', error);
            main.innerHTML = UI.renderError('Failed to load view');
        }
    },

    async loadDashboard() {
        const main = document.getElementById('main-content');
        let lastError = '';
        try {
            const [status, models] = await Promise.all([
                API.getStatus(),
                API.getModels().catch(() => ({ models: [] }))
            ]);

            // Keep for later use (model meta, etc.)
            this._lastModels = models.models || [];
            this._lastStatus = status;

            // Update the header badge immediately (not only on interval).
            UI.updateStatusIndicator(status);

            const html = UI.renderDashboard(status, models);
            main.innerHTML = html;

            // Attach event listeners
            this.attachDashboardEvents();
            // Initial summary
            this.updateOptimizationSummary();
        } catch (error) {
            console.error('Dashboard load failed:', error);
            lastError = (error && error.message) || String(error);
            main.innerHTML = UI.renderError(
                'Dashboard failed to load: ' + lastError, true
            );
        }
    },

    attachDashboardEvents() {
        // Model selector with metadata
        const modelSelect = document.getElementById('model-select');
        const modelsData = this._lastModels || [];
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                UI.updateModelMeta(e.target.value, modelsData);
                this.updateOptimizationSummary();
            });
            // Init disabled state
            UI.updateModelMeta(modelSelect.value, modelsData);
        }

        // Profile selector with custom weights handling
        document.querySelectorAll('input[name="profile"]').forEach(input => {
            input.addEventListener('change', (e) => {
                const cw = document.getElementById('custom-weights');
                if (cw) cw.classList.toggle('hidden', e.target.value !== 'custom');
                this.updateOptimizationSummary();
                // Update visual selection
                document.querySelectorAll('input[name="profile"]').forEach(r => {
                    const card = r.nextElementSibling;
                    if (card) {
                        card.className = r.checked ? 'p-4 border-2 rounded-lg border-blue-500 bg-blue-50 transition-all' : 'p-4 border-2 rounded-lg border-gray-200 hover:border-gray-300 transition-all';
                    }
                });
            });
        });

        // Custom weights validation
        ['w-gen','w-prompt','w-ctx','w-mem','w-qual','w-stab'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.addEventListener('input', () => this._validateWeights());
        });
        // Quality threshold label
        const qt = document.getElementById('quality-threshold');
        if (qt) qt.addEventListener('input', (e) => {
            document.getElementById('quality-threshold-label').textContent = e.target.value;
        });

        // Advanced settings toggles
        document.querySelectorAll('.advanced-toggle').forEach(toggle => {
            toggle.addEventListener('change', () => this.updateOptimizationSummary());
        });

        // Start optimization button with pre-run review
        const startBtn = document.getElementById('start-optimization');
        if (startBtn) {
            startBtn.addEventListener('click', () => this.showPreRunReview());
        }

        // Refresh models button
        const refreshBtn = document.getElementById('refresh-models');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadDashboard());
        }
    },

    _validateWeights() {
        const ids = ['w-gen','w-prompt','w-ctx','w-mem','w-qual','w-stab'];
        const sum = ids.reduce((s,id) => s + (parseInt(document.getElementById(id)?.value||0)), 0);
        const el = document.getElementById('weights-total');
        if (el) {
            el.textContent = `Total: ${sum}% ${sum===100 ? '✓' : '✗ must be 100%'}`;
            el.className = sum===100 ? 'text-xs mt-2 text-green-600' : 'text-xs mt-2 text-red-600';
        }
    },

    async loadHistory() {
        const { runs } = await API.getRuns();
        const html = UI.renderHistory(runs);
        document.getElementById('main-content').innerHTML = html;

        // Attach click handlers
        document.querySelectorAll('.run-row').forEach(row => {
            row.addEventListener('click', () => {
                const runId = row.dataset.runId;
                window.location.href = `/results/${runId}`;
            });
        });
    },

    async loadSettings() {
        const settings = await API.getSettings();
        const html = UI.renderSettings(settings);
        document.getElementById('main-content').innerHTML = html;

        // Attach form handlers
        const form = document.getElementById('settings-form');
        if (form) {
            form.addEventListener('submit', (e) => this.saveSettings(e));
        }

        // Test connection button
        const testBtn = document.getElementById('test-connection');
        if (testBtn) {
            testBtn.addEventListener('click', () => this.testConnection());
        }
    },

    async updateStatus() {
        try {
            const status = await API.getStatus();
            UI.updateStatusIndicator(status);
        } catch (error) {
            console.error('Status update failed:', error);
        }
    },

    updateOptimizationSummary() {
        const modelSelect = document.getElementById('model-select');
        const profile = document.querySelector('input[name="profile"]:checked');
        if (!modelSelect || !modelSelect.value) {
            const el = document.getElementById('optimization-summary');
            if (el) el.innerHTML = '<p class="text-amber-800 text-sm">Select a model to see search space review.</p>';
            return;
        }
        // Phase A/B plan estimate lives in generateOptimizationSummary
        // (staged cheap probes, not a Cartesian product).
        const adv = this.getAdvancedSettings();
        const summary = UI.generateOptimizationSummary(
            modelSelect.value,
            profile?.value || 'balanced',
            adv
        );
        const el = document.getElementById('optimization-summary');
        if (el) {
            el.innerHTML = summary;
        }
    },

    showPreRunReview() {
        const modelSelect = document.getElementById('model-select');
        const profile = document.querySelector('input[name="profile"]:checked');
        if (!modelSelect?.value) {
            UI.showToast('Please select a model', 'error');
            return;
        }
        const adv = this.getAdvancedSettings();
        const modelId = modelSelect.value;
        const prof = profile?.value || 'balanced';
        // Phase A/B estimate: cheap probes first, expensive suite on finalists.
        const est = '~16 cheap probes + ≤5 quality finalists';
        const isLarge = false;
        const html = UI.generatePreRunReview(modelId, prof, adv, est) + `
            <div class="flex justify-end gap-3 mt-6">
                <button id="review-cancel" class="btn btn-outline">Cancel</button>
                <button id="review-start" class="btn btn-success">${isLarge ? 'Continue Anyway' : 'Start Optimization'}</button>
            </div>
        `;
        UI.openModal(html, 'Optimization Review');
        document.getElementById('review-cancel')?.addEventListener('click', () => UI.closeModal());
        document.getElementById('review-start')?.addEventListener('click', () => {
            UI.closeModal();
            this.startOptimization();
        });
    },

    async startOptimization() {
        const modelSelect = document.getElementById('model-select');
        const profile = document.querySelector('input[name="profile"]:checked');
        const qualityThreshold = document.getElementById('quality-threshold');

        if (!modelSelect?.value) {
            UI.showToast('Please select a model', 'error');
            return;
        }

        const request = {
            model_id: modelSelect.value,
            profile: profile?.value || 'balanced',
            quality_threshold: parseFloat(qualityThreshold?.value || '0.97'),
            advanced_settings: this.getAdvancedSettings(),
        };

        try {
            UI.showLoading('Starting optimization...');
            const response = await API.startOptimization(request);
            UI.hideLoading();

            // Navigate to results page with WebSocket
            this.state.currentRunId = response.id;
            window.location.href = `/results/${response.id}`;
        } catch (error) {
            UI.hideLoading();
            UI.showToast(error.message || 'Failed to start optimization', 'error');
        }
    },

    getAdvancedSettings() {
        return {
            min_context: parseInt(document.getElementById('min-context')?.value || '2048'),
            max_context: parseInt(document.getElementById('max-context')?.value || '32768'),
            min_gpu_ratio: parseFloat(document.getElementById('min-gpu-ratio')?.value || '0'),
            max_gpu_ratio: parseFloat(document.getElementById('max-gpu-ratio')?.value || '1'),
            test_flash_on: document.getElementById('test-flash-on')?.checked !== false,
            test_flash_off: document.getElementById('test-flash-off')?.checked !== false,
            test_kv_gpu: document.getElementById('test-kv-gpu')?.checked !== false,
            test_kv_cpu: document.getElementById('test-kv-cpu')?.checked !== false,
            auto_batch: document.getElementById('auto-batch')?.checked !== false,
            optimize_context: document.getElementById('optimize-context')?.checked === true,
        };
    },

    async testConnection() {
        const url = document.getElementById('lm-studio-url').value;
        const btn = document.getElementById('test-connection');

        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Testing...';

        try {
            await API.testConnection(url);
            UI.showToast('Connection successful!', 'success');
        } catch (error) {
            UI.showToast(error.message || 'Connection failed', 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = 'Test Connection';
        }
    },

    async saveSettings(e) {
        e.preventDefault();
        const form = e.target;
        const formData = new FormData(form);
        const settings = {};

        for (const [key, value] of formData.entries()) {
            if (value === 'on') {
                settings[key] = true;
            } else if (value === '') {
                continue;
            } else {
                settings[key] = value;
            }
        }

        try {
            await API.updateSettings(settings);
            UI.showToast('Settings saved', 'success');
        } catch (error) {
            UI.showToast('Failed to save settings', 'error');
        }
    },
};

// Initialize on DOM ready
// readyState-safe boot: deferred modules may evaluate after DOMContentLoaded.
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => App.init());
} else {
    App.init();
}

// Export for global access
window.App = App;