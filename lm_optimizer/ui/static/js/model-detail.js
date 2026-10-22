/* Model Detail page: identity, capabilities, presets, run history, configs. */
(function () {
    "use strict";

    const API = "/api";
    const modelId = decodeURIComponent(
        window.location.pathname.replace(/^\/models\//, "").replace(/\/$/, "")
    );

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
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
    }

    function renderIdentity(model) {
        const q = model.quantization || {};
        const rows = [
            ["Model", model.id],
            ["Display name", model.name || model.display_name],
            ["Architecture", model.architecture],
            ["Quantization", q.name],
            ["Bits / weight", q.bits_per_weight],
            ["Context limit", model.max_context_length || model.context_limit],
            ["Loaded", model.loaded_instances_count || (model.loaded ? "yes" : "no")],
        ];
        el("md-title").textContent = model.name || model.display_name || model.id;
        el("md-identity").innerHTML = rows
            .map(function (r) {
                return (
                    '<div class="flex justify-between border-b border-gray-100 py-1">' +
                    "<span class='text-gray-500'>" + esc(r[0]) + "</span>" +
                    "<span class='font-mono text-xs'>" + esc(r[1] === undefined ? "-" : r[1]) + "</span>" +
                    "</div>"
                );
            })
            .join("");
    }

    function renderCapabilities(model) {
        const caps = model.capabilities || {};
        const rows = [];
        rows.push(["Vision", caps.vision ? "yes" : "no"]);
        rows.push(["Tool use", caps.trained_for_tool_use ? "yes" : "no"]);
        if (caps.reasoning) {
            const options = (caps.reasoning.allowed_options || []).join(", ");
            rows.push(["Reasoning", options || "yes"]);
            rows.push(["Default reasoning", caps.reasoning.default]);
        }
        el("md-capabilities").innerHTML = rows
            .map(function (r) {
                return (
                    '<div class="flex justify-between border-b border-gray-100 py-1">' +
                    "<span class='text-gray-500'>" + esc(r[0]) + "</span>" +
                    "<span class='font-mono text-xs'>" + esc(r[1]) + "</span>" +
                    "</div>"
                );
            })
            .join("");
    }

    function renderPresets(payload) {
        const presets = payload.presets || [];
        if (!presets.length) {
            el("md-presets").innerHTML = "<p>No presets saved for this model.</p>";
            return;
        }
        el("md-presets").innerHTML = presets
            .map(function (p) {
                return (
                    '<div class="border-b border-gray-100 py-1">' +
                    "<span class='font-medium'>" + esc(p.profile) + "</span>" +
                    "<span class='text-xs text-gray-500'> &middot; " +
                    esc((p.config && p.config.context_length) || "-") + " ctx</span></div>"
                );
            })
            .join("");
    }

    function renderRuns(payload) {
        const runs = payload.runs || payload;
        if (!Array.isArray(runs) || !runs.length) {
            el("md-runs").innerHTML = "<p>No runs for this model.</p>";
            return;
        }
        let html = "<table class='w-full text-sm'><thead><tr class='text-left text-gray-500'>";
        html += "<th class='py-1'>Run</th><th class='py-1'>Status</th><th class='py-1'>Started</th>";
        html += "<th class='py-1'>Configs</th><th class='py-1'></th></tr></thead><tbody>";
        runs.slice(0, 20).forEach(function (r) {
            html += "<tr class='border-t border-gray-100'>";
            html += "<td class='py-1 font-mono text-xs'>" + esc(String(r.id).slice(0, 8)) + "</td>";
            html += "<td class='py-1'>" + esc(r.status) + "</td>";
            html += "<td class='py-1 text-xs'>" + esc((r.started_at || r.created_at || "").slice(0, 19)) + "</td>";
            html += "<td class='py-1'>" + esc(r.config_count === undefined ? "-" : r.config_count) + "</td>";
            html += "<td class='py-1'><a class='underline text-blue-600' href='/results/" + esc(r.id) + "'>open</a></td>";
            html += "</tr>";
        });
        html += "</tbody></table>";
        el("md-runs").innerHTML = html;
    }

    async function loadConfigs(runId) {
        if (!runId) {
            el("md-configs").innerHTML = "<p>No run to read.</p>";
            return;
        }
        try {
            const payload = await getJSON(API + "/runs/" + encodeURIComponent(runId) + "/configurations");
            const configs = payload.configurations || payload;
            if (!Array.isArray(configs) || !configs.length) {
                el("md-configs").innerHTML = "<p>No configurations recorded.</p>";
                return;
            }
            let html = "<table class='w-full text-sm'><thead><tr class='text-left text-gray-500'>";
            html += "<th class='py-1'>Status</th><th class='py-1'>Ctx</th><th class='py-1'>tok/s</th>";
            html += "<th class='py-1'>TTFT</th><th class='py-1'>Score</th></tr></thead><tbody>";
            configs.slice(0, 50).forEach(function (c) {
                const metrics = c.metrics || [];
                const speeds = metrics
                    .filter(function (m) { return m.category === "speed" && m.success; })
                    .map(function (m) { return m.generation_tok_s; })
                    .filter(function (v) { return typeof v === "number"; });
                const median = speeds.length
                    ? speeds.slice().sort(function (a, b) { return a - b; })[Math.floor(speeds.length / 2)]
                    : null;
                html += "<tr class='border-t border-gray-100'>";
                html += "<td class='py-1'>" + esc(c.status) + "</td>";
                html += "<td class='py-1'>" + esc(c.context_length) + "</td>";
                html += "<td class='py-1 font-mono'>" + fmt(median, 2) + "</td>";
                html += "<td class='py-1 font-mono'>" + fmt(c.estimated_ttft_ms, 0) + " ms</td>";
                html += "<td class='py-1 font-mono'>" + fmt(c.score, 3) + "</td>";
                html += "</tr>";
            });
            html += "</tbody></table>";
            el("md-configs").innerHTML = html;
        } catch (err) {
            el("md-configs").innerHTML =
                "<p class='text-red-600'>Failed: " + esc(err.message) + "</p>";
        }
    }

    async function boot() {
        try {
            const model = await getJSON(API + "/models/" + encodeURIComponent(modelId));
            renderIdentity(model);
            renderCapabilities(model);
        } catch (err) {
            el("md-identity").textContent = "Failed: " + err.message;
        }

        try {
            renderPresets(await getJSON(API + "/presets?model_id=" + encodeURIComponent(modelId)));
        } catch (err) {
            el("md-presets").textContent = "Failed: " + err.message;
        }

        try {
            const payload = await getJSON(API + "/runs?model_id=" + encodeURIComponent(modelId) + "&limit=20");
            renderRuns(payload);
            const runs = payload.runs || payload;
            if (Array.isArray(runs) && runs.length) {
                await loadConfigs(runs[0].id);
            } else {
                el("md-configs").innerHTML = "<p>No configurations recorded.</p>";
            }
        } catch (err) {
            el("md-runs").textContent = "Failed: " + err.message;
            el("md-configs").innerHTML = "<p>No configurations recorded.</p>";
        }

        const button = el("md-run-optimize");
        if (button) {
            button.addEventListener("click", function () {
                window.location.href = "/?optimize=" + encodeURIComponent(modelId);
            });
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot);
    } else {
        boot();
    }
})();
