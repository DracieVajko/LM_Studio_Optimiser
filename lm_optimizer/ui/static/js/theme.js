// Theme manager: OS-following dark mode with manual override (persisted).
// Plain script (no modules): runs in <head> before paint to avoid flashing.
(function () {
    var KEY = 'lm-theme'; // 'dark' | 'light' | null (null = follow OS)

    function systemDark() {
        try {
            return window.matchMedia('(prefers-color-scheme: dark)').matches;
        } catch (e) {
            return false;
        }
    }

    function current() {
        try {
            return localStorage.getItem(KEY);
        } catch (e) {
            return null;
        }
    }

    function apply() {
        var saved = current();
        var dark = saved ? saved === 'dark' : systemDark();
        var changed = document.documentElement.classList.contains('dark') !== dark;
        document.documentElement.classList.toggle('dark', dark);
        if (changed) {
            try { window.dispatchEvent(new Event('lm-theme-changed')); } catch (e) {}
        }
        document.querySelectorAll('.theme-toggle').forEach(function (b) {
            b.textContent = dark ? '☀️' : '🌙';
            b.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
            b.title = dark ? 'Light mode' : 'Dark mode';
        });
    }

    window.__setTheme = function (mode) {
        try {
            if (mode) localStorage.setItem(KEY, mode);
            else localStorage.removeItem(KEY);
        } catch (e) {}
        apply();
    };

    document.addEventListener('DOMContentLoaded', function () {
        apply();
        document.querySelectorAll('.theme-toggle').forEach(function (b) {
            b.addEventListener('click', function () {
                var isDark = document.documentElement.classList.contains('dark');
                window.__setTheme(isDark ? 'light' : 'dark');
            });
        });
    });
    apply();

    try {
        var mq = window.matchMedia('(prefers-color-scheme: dark)');
        var onChange = function () { if (!current()) apply(); };
        if (mq.addEventListener) mq.addEventListener('change', onChange);
        else if (mq.addListener) mq.addListener(onChange);
    } catch (e) {}
})();
