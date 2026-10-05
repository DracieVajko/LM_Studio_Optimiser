// LM Studio Auto Optimizer - Deep Benchmark Page
// Leaderboard + per-model prompt/output/thinking preview. Outputs over
// 50000 chars paginate with a note; the full log stays in the exported .md.

import { API } from './api.js?v=6';
import { UI } from './ui.js?v=6';

const PREVIEW_PAGE_CHARS = 50000;

const DeepPage = {
    state: {
        runs: [],
        runId: null,
        detail: null,
        model: null,
        task: null,
        tab: 'output',
        page: 0,
    },

    async init() {
        UI.init();
        const ok = await this.loadRuns();
        if (!ok) return; // empty state or error card already rendered; never a spinner
        this.render();
        this.attachEvents();
    },

    async loadRuns() {
        const main = document.getElementById('main-content');
        try {
            const resp = await API.getDeepRuns();
            this.state.runs = (resp && resp.runs) || [];
            if (!this.state.runs.length) {
                main.innerHTML = '<div class="card"><div class="card-body text-center py-12">'
                    + '<h3 class="text-lg font-medium text-gray-900">No deep batches yet.</h3>'
                    + '<p class="text-sm text-gray-500 mt-1">Run the deep benchmark from the CLI first.</p>'
                    + '<div class="mt-4 flex justify-center gap-2">'
                    + '<a class="btn btn-outline" href="/history">Back to History</a>'
                    + '</div></div></div>';
                return false;
            }
            this.state.runId = this.state.runs[0].id;
            return await this.loadDetail();
        } catch (error) {
            console.error('Failed to load deep runs:', error);
            if (main) main.innerHTML = this.renderError(
                (error && error.status) || 'network/timeout',
                (error && error.message) || String(error));
            return false;
        }
    },

    async loadDetail() {
        const main = document.getElementById('main-content');
        try {
            this.state.detail = await API.getDeepRun(this.state.runId);
            const lb = (this.state.detail && this.state.detail.leaderboard) || [];
            this.state.model = lb.length ? lb[0].model : null;
            this.resetPreview();
            return true;
        } catch (error) {
            console.error('Failed to load deep batch:', error);
            if (main) main.innerHTML = this.renderError(
                (error && error.status) || 'network/timeout',
                (error && error.message) || String(error));
            return false;
        }
    },

    resetPreview() {
        const tasks = this.previewTasks();
        this.state.task = tasks.length ? tasks[0].name : null;
        this.state.tab = 'output';
        this.state.page = 0;
    },

    previewTasks() {
        const d = this.state.detail;
        if (!d || !d.models || !this.state.model) return [];
        const m = d.models[this.state.model];
        return (m && m.preview) || [];
    },

    currentTask() {
        const tasks = this.previewTasks();
        return tasks.find((t) => t.name === this.state.task) || tasks[0] || null;
    },

    renderError(status, reason) {
        return '<div class="card"><div class="card-body text-center py-12">'
            + '<h3 class="text-lg font-medium text-gray-900">Unable to load deep leaderboard.</h3>'
            + '<p class="mt-1 text-sm text-gray-500">HTTP status: ' + this.esc(String(status)) + '</p>'
            + '<p class="mt-1 text-sm text-gray-500">Reason: ' + this.esc(String(reason).slice(0, 300)) + '</p>'
            + '<div class="mt-4 flex justify-center gap-2">'
            + '<button class="btn btn-outline" onclick="window.location.reload()">Retry</button>'
            + '<a class="btn btn-outline" href="/history">Back to History</a>'
            + '</div></div></div>';
    },

    esc(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    },

    fmtScore(v) {
        if (v === null || v === undefined) return 'n/a';
        const n = Number(v);
        return Number.isFinite(n) ? n.toFixed(3) : 'n/a';
    },

    render() {
        const main = document.getElementById('main-content');
        const d = this.state.detail;
        main.innerHTML = '<div class="space-y-8">'
            + this.renderHeader()
            + this.renderLeaderboard((d && d.leaderboard) || [])
            + this.renderPreview()
            + '</div>';
        this.highlightSelection();
    },

    renderHeader() {
        const options = this.state.runs.map((r) => '<option value="' + this.esc(r.id) + '"'
            + (r.id === this.state.runId ? ' selected' : '') + '>'
            + this.esc(r.id) + ' (' + (r.completed || 0) + '/' + (r.models || 0) + ' completed)</option>').join('');
        const exportUrl = this.state.detail && this.state.detail.export_url
            ? this.state.detail.export_url : '#';
        return '<div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">'
            + '<div><h1 class="text-3xl font-bold text-gray-900">Deep Benchmark</h1>'
            + '<p class="text-gray-500 mt-1">Long-task leaderboard with full output preview</p></div>'
            + '<div class="flex gap-2 items-center">'
            + '<select id="deep-run-select" class="form-input form-select">' + options + '</select>'
            + '<a class="btn btn-outline" href="' + this.esc(exportUrl) + '">Export combined .md</a>'
            + '<a class="btn btn-outline" href="/history">History</a>'
            + '</div></div>';
    },

    renderLeaderboard(lb) {
        if (!lb.length) {
            return '<div class="card"><div class="card-body text-center py-8">'
                + '<p class="text-sm text-gray-500">No models recorded for this batch.</p></div></div>';
        }
        const rows = lb.map((r) => '<tr class="' + (r.verdict === 'winner' ? 'bg-green-50' : '') + '">'
            + '<td class="font-mono">' + this.esc(r.model) + '</td>'
            + '<td class="text-center font-mono">' + this.esc(String(r.gen_tok_s)) + '</td>'
            + '<td class="text-center font-mono">' + this.esc(this.fmtScore(r.quality)) + '</td>'
            + '<td class="text-center font-mono">' + this.esc(String(r.thinking_chars)) + '</td>'
            + '<td class="text-center font-mono">' + this.esc(String(r.elapsed_s)) + '</td>'
            + '<td class="text-center">' + this.esc(r.verdict || r.status || '') + '</td>'
            + '</tr>').join('');
        return '<div class="card"><div class="card-header"><h3 class="font-semibold">Leaderboard '
            + '(score, then gen tok/s, then elapsed)</h3></div>'
            + '<div class="card-body overflow-x-auto"><table class="w-full text-sm">'
            + '<thead><tr class="text-gray-500">'
            + '<th class="text-left">Model</th><th>Gen tok/s</th><th>Quality</th>'
            + '<th>Thinking chars</th><th>Elapsed s</th><th>Verdict</th>'
            + '</tr></thead><tbody>' + rows + '</tbody></table></div></div>';
    },

    renderPreview() {
        const d = this.state.detail;
        if (!d || !this.state.model || !d.models[this.state.model]) {
            return '<div class="card"><div class="card-body text-center py-8">'
                + '<p class="text-sm text-gray-500">Select a model to preview.</p></div></div>';
        }
        const lb = d.leaderboard || [];
        const modelBtns = lb.map((r) => '<button class="btn btn-sm deep-model-btn" data-model="'
            + this.esc(r.model) + '">' + this.esc(r.model) + '</button>').join(' ');
        const tasks = this.previewTasks();
        const taskBtns = tasks.map((t) => '<button class="btn btn-sm deep-task-btn" data-task="'
            + this.esc(t.name) + '">' + this.esc(t.name) + '</button>').join(' ');
        const tabs = ['output', 'thinking', 'prompt'].map((t) => '<button class="btn btn-sm deep-tab-btn" data-tab="'
            + t + '">' + t + '</button>').join(' ');
        return '<div class="card"><div class="card-header"><h3 class="font-semibold">Per-model preview</h3></div>'
            + '<div class="card-body space-y-4">'
            + '<div class="flex flex-wrap gap-2">' + modelBtns + '</div>'
            + '<div class="flex flex-wrap gap-2">' + taskBtns + '</div>'
            + '<div class="flex flex-wrap gap-2">' + tabs + '</div>'
            + '<div id="deep-preview-body">' + this.renderPreviewBody() + '</div>'
            + '</div></div>';
    },

    renderPreviewBody() {
        const t = this.currentTask();
        if (!t) return '<p class="text-sm text-gray-500">No preview for this model.</p>';
        const text = t[this.state.tab] || '';
        const pages = Math.max(1, Math.ceil(text.length / PREVIEW_PAGE_CHARS));
        const page = Math.min(this.state.page, pages - 1);
        const chunk = text.slice(page * PREVIEW_PAGE_CHARS, (page + 1) * PREVIEW_PAGE_CHARS);
        const note = pages > 1
            ? '<p class="text-xs text-gray-500 mt-2">Output paginated into ' + pages
              + ' pages of ' + PREVIEW_PAGE_CHARS + ' chars (page ' + (page + 1) + ' of ' + pages
              + '; full log in exported .md).</p>'
              + '<div class="flex gap-2 mt-2"><button class="btn btn-sm btn-outline" id="deep-prev"'
              + (page === 0 ? ' disabled' : '') + '>Prev</button>'
              + '<button class="btn btn-sm btn-outline" id="deep-next"'
              + (page >= pages - 1 ? ' disabled' : '') + '>Next</button></div>'
            : '';
        return '<pre class="bg-gray-900 text-green-300 rounded p-3 overflow-x-auto whitespace-pre-wrap text-xs"'
            + ' style="max-height: 32rem; overflow-y: auto;">' + this.esc(chunk) + '</pre>' + note;
    },

    highlightSelection() {
        const btns = document.querySelectorAll('.deep-model-btn');
        btns.forEach((b) => {
            if (b.getAttribute('data-model') === this.state.model) b.classList.add('btn-primary');
        });
        document.querySelectorAll('.deep-task-btn').forEach((b) => {
            if (b.getAttribute('data-task') === this.state.task) b.classList.add('btn-primary');
        });
        document.querySelectorAll('.deep-tab-btn').forEach((b) => {
            if (b.getAttribute('data-tab') === this.state.tab) b.classList.add('btn-primary');
        });
    },

    refreshPreview() {
        const body = document.getElementById('deep-preview-body');
        if (body) body.innerHTML = this.renderPreviewBody();
        this.highlightSelection();
        const prev = document.getElementById('deep-prev');
        const next = document.getElementById('deep-next');
        if (prev) prev.addEventListener('click', () => {
            this.state.page = Math.max(0, this.state.page - 1);
            this.refreshPreview();
        });
        if (next) next.addEventListener('click', () => {
            this.state.page += 1;
            this.refreshPreview();
        });
    },

    attachEvents() {
        const sel = document.getElementById('deep-run-select');
        if (sel) sel.addEventListener('change', async (e) => {
            this.state.runId = e.target.value;
            this.state.page = 0;
            const main = document.getElementById('main-content');
            main.innerHTML = '<div class="flex justify-center py-12"><div class="spinner"></div></div>';
            const ok = await this.loadDetail();
            if (!ok) return;
            this.render();
            this.attachEvents();
        });
        document.querySelectorAll('.deep-model-btn').forEach((b) => {
            b.addEventListener('click', () => {
                this.state.model = b.getAttribute('data-model');
                this.resetPreview();
                this.render();
                this.attachEvents();
            });
        });
        document.querySelectorAll('.deep-task-btn').forEach((b) => {
            b.addEventListener('click', () => {
                this.state.task = b.getAttribute('data-task');
                this.state.page = 0;
                this.refreshPreview();
            });
        });
        document.querySelectorAll('.deep-tab-btn').forEach((b) => {
            b.addEventListener('click', () => {
                this.state.tab = b.getAttribute('data-tab');
                this.state.page = 0;
                this.refreshPreview();
            });
        });
        this.refreshPreview();
    },
};

// Initialize
// readyState-safe boot: deferred modules may evaluate after DOMContentLoaded.
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => DeepPage.init());
} else {
    DeepPage.init();
}
