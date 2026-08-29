// LM Studio Auto Optimizer - API Client

export const API = {
    baseURL: '/api',

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        };

        if (config.body && typeof config.body === 'object') {
            config.body = JSON.stringify(config.body);
        }

        const response = await fetch(url, config);

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Request failed' }));
            throw new Error(error.detail || `HTTP ${response.status}`);
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
        const response = await fetch(`${this.baseURL}/runs/${runId}/export?format=${format}`);
        if (!response.ok) throw new Error('Export failed');
        return response.blob();
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