// LM Studio Auto Optimizer - Charts

export const Charts = {
    colors: {
        primary: '#2563eb',
        success: '#16a34a',
        warning: '#ea580c',
        danger: '#dc2626',
        gray: '#9ca3af',
    },

    charts: new Map(),

    init() {
        // Set default Chart.js options
        Chart.defaults.font.family = "'Inter', sans-serif";
        Chart.defaults.color = '#6b7280';
        Chart.defaults.plugins.legend.display = false;
    },

    createGenerationVsContext(ctx, data) {
        return this.createScatterChart(ctx, data, {
            xKey: 'context_length',
            yKey: 'generation_tok_s',
            xLabel: 'Context Length',
            yLabel: 'Generation tok/s',
            colorKey: 'gpu_ratio',
            pointRadius: 6,
        });
    },

    createVRAMVsContext(ctx, data) {
        return this.createScatterChart(ctx, data, {
            xKey: 'context_length',
            yKey: 'peak_vram_gb',
            xLabel: 'Context Length',
            yLabel: 'VRAM (GB)',
            colorKey: 'gpu_ratio',
            pointRadius: 6,
        });
    },

    createSpeedVsGPURatio(ctx, data) {
        return this.createScatterChart(ctx, data, {
            xKey: 'gpu_ratio',
            yKey: 'generation_tok_s',
            xLabel: 'GPU Ratio',
            yLabel: 'Generation tok/s',
            colorKey: 'context_length',
            pointRadius: 6,
        });
    },

    createQualityVsSpeed(ctx, data) {
        return this.createScatterChart(ctx, data, {
            xKey: 'quality',
            yKey: 'generation_tok_s',
            xLabel: 'Quality Score',
            yLabel: 'Generation tok/s',
            colorKey: 'context_length',
            pointRadius: 6,
        });
    },

    createParetoFrontier(ctx, allData, paretoData) {
        const chart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'All Configurations',
                        data: allData.map(d => ({
                            x: d.context_length,
                            y: d.generation_tok_s,
                            config: d,
                        })),
                        backgroundColor: this.hexToRgba(this.colors.gray, 0.3),
                        borderColor: this.colors.gray,
                        pointRadius: 5,
                        pointHoverRadius: 7,
                    },
                    {
                        label: 'Pareto Frontier',
                        data: paretoData.map(d => ({
                            x: d.context_length,
                            y: d.generation_tok_s,
                            config: d,
                        })),
                        backgroundColor: this.hexToRgba(this.colors.success, 0.5),
                        borderColor: this.colors.success,
                        pointRadius: 8,
                        pointStyle: 'triangle',
                        pointHoverRadius: 10,
                    },
                ],
            },
            options: this.getScatterOptions('Context Length', 'Generation tok/s', true),
        });

        // Add click handler for tooltips
        ctx.canvas.onclick = (event) => {
            const points = chart.getElementsAtEventForMode(event, 'nearest', { intersect: true }, true);
            if (points.length) {
                const point = points[0];
                const dataset = chart.data.datasets[point.datasetIndex];
                const config = dataset.data[point.index].config;
                this.showConfigTooltip(config, event);
            }
        };

        this.charts.set(ctx.canvas.id, chart);
        return chart;
    },

    createScatterChart(ctx, data, options) {
        const {
            xKey, yKey, xLabel, yLabel, colorKey, pointRadius = 6,
        } = options;

        // Normalize color values for gradient
        const colorValues = data.map(d => d[colorKey]).filter(v => v != null);
        const minColor = Math.min(...colorValues);
        const maxColor = Math.max(...colorValues);

        const datasets = [{
            label: '',
            data: data.map(d => ({
                x: d[xKey],
                y: d[yKey],
                config: d,
                colorValue: d[colorKey],
            })),
            backgroundColor: data.map(d => this.getColorForValue(d[colorKey], minColor, maxColor)),
            borderColor: data.map(d => this.getColorForValue(d[colorKey], minColor, maxColor, 1)),
            pointRadius,
            pointHoverRadius: pointRadius + 2,
        }];

        const chart = new Chart(ctx, {
            type: 'scatter',
            data: { datasets },
            options: this.getScatterOptions(xLabel, yLabel),
        });

        // Click handler
        ctx.canvas.onclick = (event) => {
            const points = chart.getElementsAtEventForMode(event, 'nearest', { intersect: true }, true);
            if (points.length) {
                const point = points[0];
                const config = chart.data.datasets[0].data[point.index].config;
                this.showConfigTooltip(config, event);
            }
        };

        this.charts.set(ctx.canvas.id, chart);
        return chart;
    },

    getScatterOptions(xLabel, yLabel, showLegend = false) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: showLegend },
                tooltip: {
                    enabled: false, // We use custom tooltips
                },
            },
            scales: {
                x: {
                    title: { display: true, text: xLabel, color: '#374151' },
                    grid: { color: '#e5e7eb' },
                    ticks: { color: '#6b7280' },
                },
                y: {
                    title: { display: true, text: yLabel, color: '#374151' },
                    grid: { color: '#e5e7eb' },
                    ticks: { color: '#6b7280' },
                    beginAtZero: true,
                },
            },
            interaction: {
                mode: 'nearest',
                intersect: true,
            },
        };
    },

    getColorForValue(value, min, max, alpha = 0.6) {
        if (value == null || min === max) {
            return this.hexToRgba(this.colors.primary, alpha);
        }

        const normalized = (value - min) / (max - min);
        // Blue to Green gradient
        const r = Math.round(37 * (1 - normalized) + 22 * normalized);
        const g = Math.round(99 * (1 - normalized) + 163 * normalized);
        const b = Math.round(235 * (1 - normalized) + 74 * normalized);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    },

    hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    },

    showConfigTooltip(config, event) {
        // Remove existing tooltip
        const existing = document.getElementById('chart-tooltip');
        if (existing) existing.remove();

        const tooltip = document.createElement('div');
        tooltip.id = 'chart-tooltip';
        tooltip.className = 'fixed z-50 bg-gray-900 text-white p-4 rounded-lg shadow-xl max-w-xs pointer-events-none';
        tooltip.style.left = `${event.clientX + 10}px`;
        tooltip.style.top = `${event.clientY + 10}px`;

        const metrics = [
            { label: 'Context', value: config.context_length },
            { label: 'GPU Ratio', value: config.gpu_ratio ? `${(config.gpu_ratio * 100).toFixed(0)}%` : 'Auto' },
            { label: 'Flash Attn', value: config.flash_attention ? 'ON' : 'OFF' },
            { label: 'KV Cache', value: config.offload_kv_cache_to_gpu ? 'GPU' : 'CPU' },
            { label: 'Batch', value: config.eval_batch_size || 'Auto' },
            { label: 'Gen tok/s', value: config.generation_tok_s ? config.generation_tok_s.toFixed(1) : '—' },
            { label: 'VRAM', value: config.peak_vram_gb ? `${config.peak_vram_gb.toFixed(1)} GB` : '—' },
            { label: 'Quality', value: config.quality_score ? config.quality_score.toFixed(3) : '—' },
        ];

        tooltip.innerHTML = `
            <div class="font-semibold text-lg mb-2 border-b border-gray-700 pb-2">Configuration</div>
            <div class="grid grid-cols-2 gap-2 text-sm">
                ${metrics.map(m => `
                    <div class="text-gray-400">${m.label}</div>
                    <div class="font-mono text-white">${m.value}</div>
                `).join('')}
            </div>
        `;

        document.body.appendChild(tooltip);

        // Remove on click outside
        setTimeout(() => {
            document.addEventListener('click', function removeTooltip(e) {
                if (!tooltip.contains(e.target)) {
                    tooltip.remove();
                    document.removeEventListener('click', removeTooltip);
                }
            });
        }, 0);
    },

    destroyChart(canvasId) {
        const chart = this.charts.get(canvasId);
        if (chart) {
            chart.destroy();
            this.charts.delete(canvasId);
        }
    },

    destroyAll() {
        for (const [id, chart] of this.charts) {
            chart.destroy();
        }
        this.charts.clear();
    },
};

// Initialize on load
document.addEventListener('DOMContentLoaded', () => Charts.init());