// LM Studio Auto Optimizer - API Client

export const API = {
    baseURL: '/api',
    // Every async operation has a timeout: never loading forever (§15).
    timeoutMs: 12000,
    // In-flight GET dedup: concurrent identical polls share one fetch, so a
    // runaway loop (or N tabs in one page context) cannot multiply traffic.
    _inflight: new Map(),

    async request(endpoint, options = {}) {
        const method = (options.method || 'GET').toUpperCase();
        const key = method === 'GET' ? `${method} ${endpoint}` : null;
        if (key && this._inflight.has(key)) return this._inflight.get(key);
        const pending = this._requestInner(endpoint, options).finally(() => {
            if (key && this._inflight.get(key) === pending) this._inflight.delete(key);
        });
        if (key) this._inflight.set(key, pending);
        return pending;
    },

    async _requestInner(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const timeoutMs = options.timeoutMs || this.timeoutMs;
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
            signal: controller.signal,
        };
        delete config.timeoutMs;

        if (config.body && typeof config.body === 'object') {
            config.body = JSON.stringify(config.body);
        }

        let response;
        try {
            response = await fetch(url, config);
        } catch (error) {
            if (error && error.name === 'AbortError') {
                throw new Error(`Request timed out after ${timeoutMs}ms: ${endpoint}`);
            }
            throw new Error(`Network error: ${(error && error.message) || error}`);
        } finally {
            clearTimeout(timer);
        }

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Request failed' }));
            const err = new Error(error.detail || `HTTP ${response.status}`);
            err.status = response.status;
            throw err;
        }

        return response.json();
    },

    // Status
    async getStatus() {
        return this.request('/status');
    },

    async testConnection(url) {
        return this.request('/connect', {
            method: 'POST',
            body: { url },
        });
    },

    // Models
    async getModels() {
        return this.request('/models');
    },

    async getModel(modelId) {
        return this.request(`/models/${modelId}`);
    },

    // Optimization
    async startOptimization(request) {
        return this.request('/optimize', {
            method: 'POST',
            body: request,
        });
    },

    async pauseOptimization(runId) {
        return this.request(`/optimize/${runId}/pause`, { method: 'POST' });
    },

    async resumeOptimization(runId) {
        return this.request(`/optimize/${runId}/resume`, { method: 'POST' });
    },

    async cancelOptimization(runId) {
        return this.request(`/optimize/${runId}/cancel`, { method: 'POST' });
    },

    async getOptimizationProgress(runId) {
        return this.request(`/optimize/${runId}/progress`);
    },

    async resumeRun(runId) {
        return this.request(`/optimize/${runId}/resume-run`, { method: 'POST' });
    },

    async getCheckpoints() {
        return this.request('/checkpoints');
    },

    // Runs
    async getRuns(limit = 50, offset = 0, modelId = null) {
        const params = new URLSearchParams({ limit, offset });
        if (modelId) params.append('model_id', modelId);
        return this.request(`/runs?${params}`);
    },

    async getRun(runId) {
        return this.request(`/runs/${runId}`);
    },

    async getConfigurations(runId) {
        return this.request(`/runs/${runId}/configurations`);
    },

    async getConfiguration(runId, configId) {
        return this.request(`/runs/${runId}/configurations/${configId}`);
    },

    async getParetoFrontier(runId) {
        return this.request(`/runs/${runId}/pareto`);
    },

    async exportRun(runId, format = 'json') {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), this.timeoutMs);
        try {
            const response = await fetch(`${this.baseURL}/runs/${runId}/export?format=${format}`, { signal: controller.signal });
            if (!response.ok) {
                const err = new Error('Export failed');
                err.status = response.status;
                throw err;
            }
            return response.blob();
        } catch (error) {
            if (error && error.name === 'AbortError') {
                throw new Error(`Request timed out after ${this.timeoutMs}ms: export`);
            }
            throw error;
        } finally {
            clearTimeout(timer);
        }
    },

    // Apply Configuration
    async applyConfiguration(request) {
        return this.request('/apply', {
            method: 'POST',
            body: request,
        });
    },

    async restorePrevious(modelId) {
        return this.request(`/restore/${modelId}`, { method: 'POST' });
    },

    // Presets
    async getPresets(modelId = null) {
        const params = modelId ? `?model_id=${modelId}` : '';
        return this.request(`/presets${params}`);
    },

    async savePreset(preset) {
        return this.request('/presets', {
            method: 'POST',
            body: preset,
        });
    },

    async deletePreset(presetId) {
        return this.request(`/presets/${presetId}`, { method: 'DELETE' });
    },

    async applyPreset(presetId) {
        return this.request(`/presets/${presetId}/apply`, { method: 'POST' });
    },

    // Sandbox duels
    async getDuels(limit = 50, offset = 0) {
        return this.request(`/duels?limit=${limit}&offset=${offset}`);
    },

    async getDuel(jobId) {
        return this.request(`/duels/${jobId}`);
    },

    async deleteDuel(jobId) {
        return this.request(`/duels/${jobId}`, { method: 'DELETE' });
    },

    async deleteRun(runId) {
        return this.request(`/runs/${runId}`, { method: 'DELETE' });
    },

    async abandonRun(runId) {
        return this.request(`/runs/${runId}/abandon`, { method: 'POST' });
    },

    // Settings
    async getSettings() {
        return this.request('/settings');
    },

    async updateSettings(settings) {
        return this.request('/settings', {
            method: 'PUT',
            body: settings,
        });
    },
};