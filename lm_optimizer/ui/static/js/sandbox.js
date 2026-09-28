// Model sandbox: text / html / scene duels. POST /api/sandbox, poll, render.
(function () {
    var kind = 'text';
    function el(id) { return document.getElementById(id); }
    function out(html) { el('sb-out').innerHTML = html; }
    function esc(s) {
        return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
        });
    }

    document.querySelectorAll('.sb-kind').forEach(function (b) {
        b.addEventListener('click', function () {
            kind = b.dataset.kind;
            document.querySelectorAll('.sb-kind').forEach(function (x) {
                x.classList.toggle('btn-primary', x === b);
                x.classList.toggle('btn-outline', x !== b);
            });
            renderPresets();
        });
    });
    renderPresets();
    document.querySelector('.sb-kind').classList.add('btn-primary');
    document.querySelector('.sb-kind').classList.remove('btn-outline');

    var PRESETS = {
        text: [
            { label: 'Hash-table explainer', prompt: 'Write a concise explanation of how a hash table works in 3-4 sentences.' },
            { label: 'find_duplicates coding', prompt: 'Write a Python function `find_duplicates(nums: list[int]) -> list[int]` that returns all duplicate integers in a list: O(n) time, O(1) extra space (excluding output), handle negatives, ascending order. Only the function definition with docstring.' },
        ],
        html: [
            { label: 'Coffee landing page', prompt: 'Build a small responsive landing page for a fictional coffee shop: hero header, three feature cards, footer. Single self-contained HTML file, embedded CSS, clean modern look.' },
            { label: 'Todo app', prompt: 'Build a functional single-file todo app: add, toggle and delete tasks with localStorage persistence. Embedded CSS and vanilla JavaScript, no frameworks.' },
        ],
        scene: [
            { label: 'Skyblock demo', prompt: null }, // filled below (full spec)
            { label: 'Simple spinner', prompt: 'Build a single self-contained HTML file with one slowly rotating 3D cube (Three.js via CDN), dark background, nothing else.' },
        ],
    };
    PRESETS.scene[0].prompt = 'Act as an expert frontend engineer, WebGL specialist, and 3D game developer. '
        + 'Generate a complete, production-grade, single-file HTML page containing a mini-3D voxel game render '
        + 'inspired by "Minecraft Skyblock" using Three.js. Self-contained in one code block, no placeholders, '
        + 'no omissions. HTML5 boilerplate, Three.js via reliable public CDN (cdnjs or unpkg), CSS reset, '
        + 'canvas 100% viewport, dark background. Floating island of individual colored voxels '
        + '(BoxGeometry + MeshStandardMaterial or Basic for performance): green grass top layer, brown/grey '
        + 'dirt-stone under-layers, semi-transparent blue water in a carved recess, minimalist voxel oak tree '
        + '(brown trunk, green foliage) on the grass. Camera: OrbitControls from CDN if safe, else smooth '
        + 'automatic 360-degree orbit. Lights: AmbientLight plus DirectionalLight with shadows. '
        + 'Performance: grouped meshes, requestAnimationFrame 60 FPS loop, dynamic resize handling, vanilla JS only. '
        + 'Output only the complete HTML code, starting directly with <!DOCTYPE html>.';

    function renderPresets() {
        var box = el('sb-presets');
        box.innerHTML = '';
        (PRESETS[kind] || []).forEach(function (p) {
            var b = document.createElement('button');
            b.className = 'btn btn-outline btn-sm';
            b.textContent = p.label;
            b.addEventListener('click', function () { el('sb-prompt').value = p.prompt; });
            box.appendChild(b);
        });
    }

    function toTextFallback(select) {
        var input = document.createElement('input');
        input.id = select.id;
        input.className = select.className;
        input.placeholder = 'model id (list failed to load)';
        select.replaceWith(input);
    }

    async function loadModels() {
        try {
            var r = await fetch('/api/models');
            if (!r.ok) throw new Error('HTTP ' + r.status);
            var j = await r.json();
            var models = j.models || [];
            if (!models.length) throw new Error('no models');
            ['sb-a', 'sb-b'].forEach(function (id) {
                var sel = el(id);
                sel.innerHTML = '';
                models.forEach(function (m, i) {
                    var o = document.createElement('option');
                    o.value = m.id;
                    o.textContent = m.name || m.id;
                    sel.appendChild(o);
                });
                sel.selectedIndex = id === 'sb-b' ? Math.min(1, models.length - 1) : 0;
            });
        } catch (e) {
            toTextFallback(el('sb-a'));
            toTextFallback(el('sb-b'));
            out('<p class="text-sm text-yellow-700">Model list failed to load — type model IDs manually. (' + esc((e && e.message) || e) + ')</p>');
        }
    }
    loadModels();

    async function poll(id) {
        for (var i = 0; i < 180; i++) {
            var r = await fetch('/api/sandbox/' + encodeURIComponent(id));
            var j = await r.json();
            if (j.status === 'done') return j;
            if (j.status === 'failed') throw new Error(j.error || 'sandbox failed');
            await new Promise(function (res) { setTimeout(res, 3000); });
        }
        throw new Error('Timed out waiting for duel result');
    }

    function fileBlock(side, label, base) {
        if (!side || !side.ok || !side.file_path) {
            return '<h4 class="font-bold">' + label + '</h4>'
                + '<p class="text-red-600 text-sm">' + esc((side && side.error) || 'failed') + '</p>';
        }
        // job_id/side mirrored into the file-serving route path
        var parts = String(side.file_path).split('/');
        var url = '/sandbox/files/' + parts[0] + '/' + parts[1] + '/index.html';
        return '<h4 class="font-bold">' + label + ' <span class="text-xs font-normal text-gray-500">'
            + esc(side.tok_s ? side.tok_s.toFixed(1) + ' tok/s' : '') + '</span></h4>'
            + '<p class="text-xs font-mono text-gray-500">results/sandbox/' + esc(side.file_path) + '</p>'
            + '<p class="mt-1"><a class="underline text-blue-600" target="_blank" rel="noopener" href="' + url + '">Open index.html</a></p>'
            + '<iframe sandbox="" src="' + url + '" class="w-full mt-2 border rounded" style="height:420px" title="' + label + ' preview"></iframe>'
            + (base ? '' : '');
    }

    el('sb-run').addEventListener('click', async function () {
        try {
            var a = (el('sb-a').value || '').trim();
            var b = (el('sb-b').value || '').trim();
            var p = (el('sb-prompt').value || '').trim();
            if (!a || !b) throw new Error('Both model A and B are required');
            if (!p) throw new Error('Prompt is required');
            out('<p class="text-sm">Duel running: A first, unload, then B...</p>');
            var r = await fetch('/api/sandbox', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_a: a, model_b: b, prompt: p, kind: kind }),
            });
            if (!r.ok) {
                var e = await r.json().catch(function () { return {}; });
                throw new Error((e && e.detail) || ('HTTP ' + r.status));
            }
            var started = await r.json();
            var done = await poll(started.job_id);
            var d = done.result || {};
            var head = '<h3 class="font-bold">Done'
                + (d.faster && d.faster !== 'draw' ? ' — ' + esc(d.faster) + ' was faster' : d.faster === 'draw' ? ' — tie' : '')
                + '</h3>';
            if (kind === 'text') {
                out(head + '<div class="grid md:grid-cols-2 gap-4 mt-2">'
                    + '<div><h4 class="font-bold">A: ' + esc(d.model_a) + '</h4>'
                    + '<p class="text-xs text-gray-500">' + esc(d.side_a.tok_s ? d.side_a.tok_s.toFixed(1) + ' tok/s' : (d.side_a.error || '')) + '</p>'
                    + '<pre class="font-mono text-xs bg-gray-50 p-2 rounded overflow-auto max-h-96 mt-1">' + esc(d.side_a.text || d.side_a.error || '') + '</pre></div>'
                    + '<div><h4 class="font-bold">B: ' + esc(d.model_b) + '</h4>'
                    + '<p class="text-xs text-gray-500">' + esc(d.side_b.tok_s ? d.side_b.tok_s.toFixed(1) + ' tok/s' : (d.side_b.error || '')) + '</p>'
                    + '<pre class="font-mono text-xs bg-gray-50 p-2 rounded overflow-auto max-h-96 mt-1">' + esc(d.side_b.text || d.side_b.error || '') + '</pre></div>'
                    + '</div>');
            } else {
                out(head + '<div class="grid md:grid-cols-2 gap-4 mt-2">'
                    + '<div>' + fileBlock(d.side_a, 'A: ' + esc(d.model_a)) + '</div>'
                    + '<div>' + fileBlock(d.side_b, 'B: ' + esc(d.model_b)) + '</div>'
                    + '</div>');
            }
        } catch (e) {
            out('<p class="text-red-600 text-sm">Error: ' + esc((e && e.message) || e) + '</p>');
        }
    });
})();
