/* Experiment Builder: mode -> model -> budget -> review -> run. */
(function () {
    "use strict";

    // Embedded in the Brewery tab: hide the duplicate nav chrome.
    if (/[?&]embed=1/.test(window.location.search)) {
        const nav = document.querySelector("nav");
        if (nav) nav.style.display = "none";
        const main = document.querySelector("main");
        if (main) main.style.paddingTop = "0.5rem";
    }

    const API = "/api";

    const MODE_HINTS = {
        full: "Speed discovery + quality suite + recovery + final validation.",
        speed: "S0 budget + planned speed probes at the pinned context. No winner.",
        quality: "One quality suite on the baseline config at the max context.",
        reasoning: "5-level reasoning sweep (off/low/medium/high/xhigh).",
        mtp: "S0 + speculative/MTP probes at the pinned context, OFF baseline included.",
    };

    const MODE_SKIP = {
        full: [],
        speed: ["quality", "recovery", "validation", "context"],
        quality: ["speed", "recovery", "validation", "context"],
        reasoning: ["speed", "recovery", "validation", "context"],
        mtp: ["quality", "recovery", "validation", "context"],
    };

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

    function mode() {
        const checked = document.querySelector('input[name="ex-mode"]:checked');
        return checked ? checked.value : "full";
    }

    function buildRequest() {
        const currentMode = mode();
        const advanced = {
            min_context: 2048,
            max_context: parseInt(el("ex-maxctx").value, 10) || 32768,
            quality_context: parseInt(el("ex-quality-ctx").value, 10) || 8192,
            min_generation_tok_s: parseFloat(el("ex-min-tps").value) || 2.0,
            skip_phases: MODE_SKIP[currentMode].slice(),
            quality_lab: el("ex-quality-lab").checked,
        };
        if (currentMode === "speed" || currentMode === "mtp") {
            advanced.speed_context = parseInt(el("ex-speedctx").value, 10) || 16384;
            advanced.speed_only = currentMode === "speed";
            advanced.mtp_test = currentMode === "mtp";
        }
        if (currentMode === "quality") {
            advanced.quality_only = true;
        }
        if (currentMode === "reasoning") {
            advanced.reasoning_sweep = true;
        }
        return {
            model_id: el("ex-model").value,
            profile: el("ex-profile").value,
            quality_threshold: parseFloat(el("ex-threshold").value) || 0.97,
            min_generation_tok_s: parseFloat(el("ex-min-tps").value) || 2.0,
            advanced_settings: advanced,
        };
    }

    function renderReview() {
        const request = buildRequest();
        const review = {
            mode: mode(),
            model: request.model_id || "(select a model)",
            profile: request.profile,
            quality_threshold: request.quality_threshold,
            advanced_settings: request.advanced_settings,
        };
        el("ex-review").textContent = JSON.stringify(review, null, 2);
        const start = el("ex-start");
        start.disabled = !request.model_id;
        el("ex-mode-hint").textContent = MODE_HINTS[mode()] || "";
        el("ex-speedctx-wrap").classList.toggle("hidden", !(mode() === "speed" || mode() === "mtp"));
    }

    async function loadModels() {
        try {
            const response = await fetch(API + "/models", { headers: { Accept: "application/json" } });
            const payload = await response.json();
            const models = payload.models || payload;
            const select = el("ex-model");
            select.innerHTML = "<option value=''>Select a model...</option>";
            (models || []).forEach(function (m) {
                const option = document.createElement("option");
                option.value = m.id;
                const q = m.quantization && m.quantization.name ? m.quantization.name : "";
                option.textContent = m.id + (q ? " (" + q + ")" : "");
                select.appendChild(option);
            });
        } catch (err) {
            el("ex-model").innerHTML = "<option value=''>Failed to load models</option>";
        }
    }

    async function startRun() {
        const request = buildRequest();
        const status = el("ex-status");
        const button = el("ex-start");
        button.disabled = true;
        status.textContent = "Starting...";
        try {
            const response = await fetch(API + "/optimize", {
                method: "POST",
                headers: { "Content-Type": "application/json", Accept: "application/json" },
                body: JSON.stringify(request),
            });
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.detail || "HTTP " + response.status);
            }
            status.innerHTML =
                "Started run <a class='underline text-blue-600' href='/results/" +
                esc(payload.id) + "'>" + esc(String(payload.id).slice(0, 8)) + "</a>.";
        } catch (err) {
            status.textContent = "Failed: " + err.message;
            button.disabled = false;
        }
    }

    function boot() {
        loadModels();
        document.querySelectorAll('input[name="ex-mode"]').forEach(function (input) {
            input.addEventListener("change", renderReview);
        });
        ["ex-model", "ex-profile", "ex-threshold", "ex-maxctx", "ex-min-tps",
         "ex-quality-ctx", "ex-speedctx", "ex-quality-lab"].forEach(
            function (id) {
                const node = el(id);
                if (node) {
                    node.addEventListener("change", renderReview);
                    node.addEventListener("input", renderReview);
                }
            }
        );
        el("ex-start").addEventListener("click", startRun);
        renderReview();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot);
    } else {
        boot();
    }
})();
