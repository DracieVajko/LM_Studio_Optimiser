/* Lab page: Recommender, Quality tasks, Duel types.
   Uses the same API base as the rest of the UI (/api prefix). */
(function () {
    "use strict";

    const API = "/api";

    function el(id) {
        return document.getElementById(id);
    }

    function esc(value) {
        return String(value === null || value === undefined ? "" : value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    function fmt(value, digits) {
        if (value === null || value === undefined) return "-";
        if (typeof value !== "number") return String(value);
        return value.toFixed(digits === undefined ? 2 : digits);
    }

    async function getJSON(url) {
        const response = await fetch(url, { headers: { Accept: "application/json" } });
        if (!response.ok) {
            throw new Error("HTTP " + response.status);
        }
        return response.json();
    }

    /* --- tabs --- */
    function initTabs() {
        const tabs = document.querySelectorAll(".lab-tab");
        tabs.forEach(function (tab) {
            tab.addEventListener("click", function () {
                tabs.forEach(function (t) { t.classList.remove("btn-primary"); });
                tab.classList.add("btn-primary");
                const name = tab.getAttribute("data-tab");
                document.querySelectorAll(".lab-panel").forEach(function (panel) {
                    panel.classList.add("hidden");
                });
                const panel = el("lab-panel-" + name);
                if (panel) panel.classList.remove("hidden");
                if (name === "tasks" && !el("lab-tasks-result").dataset.loaded) {
                    loadTasks();
                }
                if (name === "duels" && !el("lab-duels-result").dataset.loaded) {
                    loadDuelTypes();
                }
                if (name === "sandbox" && !el("lab-sandbox-result").dataset.loaded) {
                    loadSandbox();
                }
            });
        });
    }

    /* --- recommender --- */
    function renderRecommendations(payload) {
        const target = el("lab-recommend-result");
        const recs = payload.recommendations || [];
        const hw = payload.hardware || {};
        let html = "<div class='card'><div class='card-body'>";
        html += "<h3 class='font-semibold mb-2'>Detected hardware</h3>";
        html += "<p class='text-sm text-gray-500 mb-4'>";
        html += esc(hw.gpu_name || "unknown GPU");
        html += " &middot; " + fmt(hw.vram_gb, 1) + " GB VRAM";
        html += " &middot; " + esc(hw.cpu_name || "unknown CPU");
        html += " &middot; " + fmt(hw.ram_gb, 1) + " GB RAM (" + (hw.ram_channels || 1) + " channel)";
        html += "</p>";

        if (!recs.length) {
            html += "<p class='text-sm text-gray-500'>No recommendations.</p>";
        } else {
            html += "<table class='w-full text-sm'>";
            html += "<thead><tr class='text-left text-gray-500'>";
            html += "<th class='py-1'>Model</th><th class='py-1'>Size</th>";
            html += "<th class='py-1'>Fit</th><th class='py-1'>tok/s</th>";
            html += "<th class='py-1'>Evidence</th><th class='py-1'>Why</th></tr></thead><tbody>";
            recs.forEach(function (rec) {
                const m = rec.model || {};
                const label = m.quant + " " + fmt(m.total_params_b, 0) + "B";
                let evidence = "estimate";
                if (rec.local_evidence) {
                    evidence = "local" + (rec.measured_context ? " @ctx " + rec.measured_context : "");
                } else if (rec.confidence) {
                    evidence = "community (" + rec.confidence + ")";
                }
                html += "<tr class='border-t border-gray-200'>";
                html += "<td class='py-1'>" + esc(label) + "</td>";
                html += "<td class='py-1'>" + fmt(m.file_size_gb, 1) + " GB</td>";
                html += "<td class='py-1'>" + esc(rec.fit_class) + "</td>";
                html += "<td class='py-1 font-mono'>" + fmt(rec.expected_tps, 2) + "</td>";
                html += "<td class='py-1 text-xs'>" + esc(evidence) + "</td>";
                html += "<td class='py-1 text-xs text-gray-500'>" + esc((rec.reasons || []).join("; ")) + "</td>";
                html += "</tr>";
            });
            html += "</tbody></table>";
        }
        html += "</div></div>";
        target.innerHTML = html;
    }

    async function runRecommend() {
        const button = el("lab-run-recommend");
        const target = el("lab-recommend-result");
        button.disabled = true;
        target.innerHTML = "<p class='text-sm text-gray-500'>Computing...</p>";
        const params = new URLSearchParams();
        params.set("min_tps", el("lab-min-tps").value || "5");
        params.set("min_context", el("lab-min-ctx").value || "8192");
        params.set("use_case", el("lab-use-case").value || "general");
        if (el("lab-vram-only").checked) params.set("vram_only", "true");
        try {
            const payload = await getJSON(API + "/lab/recommend?" + params.toString());
            renderRecommendations(payload);
        } catch (err) {
            target.innerHTML = "<p class='text-sm text-red-600'>Failed: " + esc(err.message) + "</p>";
        } finally {
            button.disabled = false;
        }
    }

    /* --- quality tasks --- */
    async function loadTasks() {
        const target = el("lab-tasks-result");
        try {
            const payload = await getJSON(API + "/lab/quality-tasks");
            const tasks = payload.tasks || [];
            let html = "<table class='w-full text-sm'>";
            html += "<thead><tr class='text-left text-gray-500'>";
            html += "<th class='py-1'>Task</th><th class='py-1'>Category</th>";
            html += "<th class='py-1'>Validators</th><th class='py-1'>Hard gates</th></tr></thead><tbody>";
            tasks.forEach(function (task) {
                html += "<tr class='border-t border-gray-200'>";
                html += "<td class='py-1 font-mono text-xs'>" + esc(task.id) + "</td>";
                html += "<td class='py-1'>" + esc(task.category) + "</td>";
                html += "<td class='py-1 text-xs'>" + esc((task.validators || []).join(", ")) + "</td>";
                html += "<td class='py-1 text-xs'>" + esc((task.hard_gates || []).join(", ")) + "</td>";
                html += "</tr>";
            });
            html += "</tbody></table>";
            target.innerHTML = html;
            target.dataset.loaded = "1";
        } catch (err) {
            target.innerHTML = "<p class='text-sm text-red-600'>Failed: " + esc(err.message) + "</p>";
        }
    }

    /* --- duel types --- */
    async function loadDuelTypes() {
        const target = el("lab-duels-result");
        try {
            const payload = await getJSON(API + "/lab/duel-types");
            const types = payload.types || [];
            let html = "<table class='w-full text-sm'>";
            html += "<thead><tr class='text-left text-gray-500'>";
            html += "<th class='py-1'>Type</th><th class='py-1'>Tasks</th></tr></thead><tbody>";
            types.forEach(function (t) {
                html += "<tr class='border-t border-gray-200'>";
                html += "<td class='py-1 font-mono text-xs'>" + esc(t.id) + "</td>";
                html += "<td class='py-1 text-xs'>" + esc((t.tasks || []).join(", ")) + "</td>";
                html += "</tr>";
            });
            html += "</tbody></table>";
            target.innerHTML = html;
            target.dataset.loaded = "1";
        } catch (err) {
            target.innerHTML = "<p class='text-sm text-red-600'>Failed: " + esc(err.message) + "</p>";
        }
    }

    // readyState-safe boot: deferred modules may evaluate after DOMContentLoaded.
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }

    /* --- sandbox --- */
    let sandboxTasks = {};

    async function loadSandbox() {
        const taskSelect = el("lab-sandbox-task");
        const modelSelect = el("lab-sandbox-model");
        try {
            const payload = await getJSON(API + "/lab/sandbox-tasks");
            sandboxTasks = {};
            taskSelect.innerHTML = "";
            (payload.tasks || []).forEach(function (task) {
                sandboxTasks[task.id] = task;
                const option = document.createElement("option");
                option.value = task.id;
                option.textContent = task.kind + " \u2014 " + task.title;
                taskSelect.appendChild(option);
            });
            renderSandboxChecks();
        } catch (err) {
            taskSelect.innerHTML = "<option value=''>Failed to load tasks</option>";
        }
        try {
            const payload = await getJSON(API + "/models");
            const models = payload.models || payload;
            modelSelect.innerHTML = "<option value=''>Select a model...</option>";
            (models || []).forEach(function (m) {
                const option = document.createElement("option");
                option.value = m.id;
                option.textContent = m.id;
                modelSelect.appendChild(option);
            });
        } catch (err) {
            modelSelect.innerHTML = "<option value=''>Failed to load models</option>";
        }
        el("lab-sandbox-result").dataset.loaded = "1";
    }

    function renderSandboxChecks() {
        const task = sandboxTasks[el("lab-sandbox-task").value];
        if (!task) {
            el("lab-sandbox-checks").textContent = "";
            return;
        }
        const checks = (task.checks || [])
            .map(function (c) {
                return (c.critical ? "* " : "") + c.name + (c.negate ? " (must NOT match)" : "");
            })
            .join(", ");
        el("lab-sandbox-checks").textContent =
            "Acceptance criteria (* = critical): " + checks +
            " | repair attempts: " + task.max_repair_attempts +
            " | timeout: " + task.timeout_s + "s";
    }

    async function runSandbox() {
        const task = el("lab-sandbox-task").value;
        const model = el("lab-sandbox-model").value;
        const target = el("lab-sandbox-result");
        const button = el("lab-sandbox-run");
        if (!task || !model) {
            target.textContent = "Select a task and a model first.";
            return;
        }
        button.disabled = true;
        target.innerHTML = "<p class='text-gray-500'>Starting...</p>";
        try {
            const response = await fetch(API + "/lab/sandbox-run", {
                method: "POST",
                headers: { "Content-Type": "application/json", Accept: "application/json" },
                body: JSON.stringify({ task: task, model: model }),
            });
            const payload = await response.json();
            if (!response.ok) throw new Error(payload.detail || "HTTP " + response.status);
            await pollSandbox(payload.job_id);
        } catch (err) {
            target.innerHTML = "<p class='text-red-600'>Failed: " + esc(err.message) + "</p>";
        } finally {
            button.disabled = false;
        }
    }

    async function pollSandbox(jobId) {
        const target = el("lab-sandbox-result");
        for (let attempt = 0; attempt < 120; attempt++) {
            await new Promise(function (resolve) { setTimeout(resolve, 5000); });
            const payload = await getJSON(API + "/lab/sandbox-run/" + encodeURIComponent(jobId));
            if (payload.status === "completed") {
                const result = payload.result || {};
                target.innerHTML =
                    "<p><strong>Result:</strong> " +
                    (result.final_success ? "PASS" : "FAIL") +
                    " &middot; first pass: " + (result.first_pass_success ? "yes" : "no") +
                    " &middot; repairs: " + (result.repair_count || 0) +
                    " &middot; score: " + fmt(result.final_score, 3) +
                    " (penalized " + fmt(result.penalized_score, 3) + ")</p>" +
                    "<pre class='text-xs bg-gray-100 p-2 rounded mt-2'>" +
                    esc(JSON.stringify(result, null, 2)) + "</pre>";
                return;
            }
            if (payload.status === "failed") {
                target.innerHTML =
                    "<p class='text-red-600'>Failed: " + esc(payload.error || "unknown") + "</p>";
                return;
            }
            target.innerHTML = "<p class='text-gray-500'>Running (" + payload.status + ")...</p>";
        }
        target.innerHTML = "<p class='text-yellow-600'>Still running after 10 minutes; check the CLI.</p>";
    }

    function init() {
        initTabs();
        const button = el("lab-run-recommend");
        if (button) button.addEventListener("click", runRecommend);
        const sandboxButton = el("lab-sandbox-run");
        if (sandboxButton) sandboxButton.addEventListener("click", runSandbox);
        const taskSelect = el("lab-sandbox-task");
        if (taskSelect) taskSelect.addEventListener("change", renderSandboxChecks);
        runRecommend();
    }
})();
