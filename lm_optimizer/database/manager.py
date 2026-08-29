"""Database configuration and connection management."""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from lm_optimizer.config import config


class DatabaseManager:
    """Manages SQLite database connections and migrations."""

    def __init__(self, db_path: Path | None = None):
        # Prefer new data/ location, fallback to legacy config/optimizer.db for migration
        if db_path:
            self.db_path = db_path
        else:
            # Check for legacy location
            legacy = config.storage.config_dir / "optimizer.db"
            new_path = config.storage.database_path
            if legacy.exists() and not new_path.exists():
                # Migrate legacy DB
                try:
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil

                    shutil.copy2(legacy, new_path)
                except Exception:
                    pass
            self.db_path = new_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database with schema and migrations."""
        with self.get_connection() as conn:
            self._run_migrations(conn)

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Run database migrations."""
        # Create migrations table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version INTEGER NOT NULL UNIQUE,
                name TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Get applied migrations
        applied = {row["version"] for row in conn.execute("SELECT version FROM migrations")}

        migrations = [
            (1, "initial_schema", self._migration_001_initial_schema),
            (2, "add_presets", self._migration_002_presets),
            (3, "add_settings", self._migration_003_settings),
            (4, "add_benchmarks", self._migration_004_benchmarks),
            (5, "add_quality_results", self._migration_005_quality_results),
            (6, "add_optimization_correctness", self._migration_006_optimization_correctness),
            (7, "add_benchmark_params", self._migration_007_benchmark_params),
        ]

        for version, name, migration_func in migrations:
            if version not in applied:
                migration_func(conn)
                conn.execute(
                    "INSERT INTO migrations (version, name) VALUES (?, ?)", (version, name)
                )

    def _migration_001_initial_schema(self, conn: sqlite3.Connection) -> None:
        """Initial schema with runs, configurations, hardware, models."""
        # Hardware snapshots
        conn.execute("""
            CREATE TABLE hardware_snapshots (
                id TEXT PRIMARY KEY,
                os TEXT NOT NULL,
                cpu_name TEXT NOT NULL,
                cpu_cores_physical INTEGER NOT NULL,
                cpu_cores_logical INTEGER NOT NULL,
                total_ram_gb REAL NOT NULL,
                gpu_count INTEGER NOT NULL,
                gpus_json TEXT NOT NULL,
                cuda_version TEXT,
                metal_available INTEGER NOT NULL DEFAULT 0,
                vulkan_available INTEGER NOT NULL DEFAULT 0,
                detected_at TIMESTAMP NOT NULL
            )
        """)

        # Models
        conn.execute("""
            CREATE TABLE models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                architecture TEXT,
                parameter_count INTEGER,
                quantization TEXT,
                context_limit INTEGER,
                is_moe INTEGER NOT NULL DEFAULT 0,
                num_experts INTEGER,
                size_bytes INTEGER,
                metadata_json TEXT,
                first_seen TIMESTAMP NOT NULL,
                last_seen TIMESTAMP NOT NULL
            )
        """)

        # Optimization runs
        conn.execute("""
            CREATE TABLE runs (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                hardware_id TEXT NOT NULL,
                profile TEXT NOT NULL,
                profile_weights_json TEXT NOT NULL,
                quality_threshold REAL NOT NULL DEFAULT 0.97,
                status TEXT NOT NULL DEFAULT 'pending',
                stage TEXT NOT NULL DEFAULT 'discovery',
                search_space_json TEXT,
                optimizer_version TEXT NOT NULL,
                benchmark_repetitions INTEGER NOT NULL DEFAULT 3,
                validation_repetitions INTEGER NOT NULL DEFAULT 5,
                created_at TIMESTAMP NOT NULL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                duration_seconds REAL DEFAULT 0,
                error TEXT,
                FOREIGN KEY (model_id) REFERENCES models(id),
                FOREIGN KEY (hardware_id) REFERENCES hardware_snapshots(id)
            )
        """)

        # Configurations
        conn.execute("""
            CREATE TABLE configurations (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                config_json TEXT NOT NULL,
                context_length INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                metrics_json TEXT,
                quality_json TEXT,
                stability_score REAL DEFAULT 1.0,
                peak_vram_gb REAL,
                peak_ram_gb REAL,
                error TEXT,
                score REAL DEFAULT 0.0,
                tested_at TIMESTAMP,
                duration_ms REAL DEFAULT 0,
                FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
            )
        """)

        # Create indexes
        conn.execute("CREATE INDEX idx_runs_model ON runs(model_id)")
        conn.execute("CREATE INDEX idx_runs_status ON runs(status)")
        conn.execute("CREATE INDEX idx_runs_created ON runs(created_at)")
        conn.execute("CREATE INDEX idx_configs_run ON configurations(run_id)")
        conn.execute("CREATE INDEX idx_configs_status ON configurations(status)")

    def _migration_002_presets(self, conn: sqlite3.Connection) -> None:
        """Add presets table."""
        conn.execute("""
            CREATE TABLE presets (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                profile TEXT NOT NULL,
                name TEXT,
                config_json TEXT NOT NULL,
                metrics_json TEXT,
                quality_json TEXT,
                run_id TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                optimizer_version TEXT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models(id),
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
        """)
        conn.execute("CREATE INDEX idx_presets_model ON presets(model_id)")

    def _migration_003_settings(self, conn: sqlite3.Connection) -> None:
        """Add settings table."""
        conn.execute("""
            CREATE TABLE settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Insert default settings
        defaults = {
            "lm_studio_url": "http://127.0.0.1:1234",
            "web_host": "127.0.0.1",
            "web_port": "8080",
            "default_profile": "balanced",
            "default_quality_threshold": "0.97",
            "default_benchmark_repetitions": "3",
            "default_validation_repetitions": "5",
            "auto_unload_after_tests": "1",
            "timeout_seconds": "300",
        }
        for key, value in defaults.items():
            conn.execute(
                "INSERT OR IGNORE INTO settings (key, value, description) VALUES (?, ?, ?)",
                (key, value, f"Default {key}"),
            )

    def _migration_004_benchmarks(self, conn: sqlite3.Connection) -> None:
        """Add detailed benchmarks table."""
        conn.execute("""
            CREATE TABLE benchmarks (
                id TEXT PRIMARY KEY,
                config_id TEXT NOT NULL,
                test_name TEXT NOT NULL,
                category TEXT NOT NULL,
                repetition INTEGER NOT NULL,
                success INTEGER NOT NULL,
                load_time_ms REAL,
                ttft_ms REAL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                prompt_processing_ms REAL,
                generation_ms REAL,
                prompt_tok_s REAL,
                generation_tok_s REAL,
                output_text TEXT,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (config_id) REFERENCES configurations(id) ON DELETE CASCADE
            )
        """)
        conn.execute("CREATE INDEX idx_benchmarks_config ON benchmarks(config_id)")

    def _migration_005_quality_results(self, conn: sqlite3.Connection) -> None:
        """Add quality results table."""
        conn.execute("""
            CREATE TABLE quality_results (
                id TEXT PRIMARY KEY,
                config_id TEXT NOT NULL,
                test_name TEXT NOT NULL,
                overall REAL NOT NULL,
                task_completion REAL,
                factual_consistency REAL,
                format_compliance REAL,
                coding_correctness REAL,
                no_truncation REAL,
                no_malformed REAL,
                confident INTEGER NOT NULL DEFAULT 1,
                details_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (config_id) REFERENCES configurations(id) ON DELETE CASCADE
            )
        """)
        conn.execute("CREATE INDEX idx_quality_config ON quality_results(config_id)")

    def _migration_006_optimization_correctness(self, conn: sqlite3.Connection) -> None:
        """Add hardware-agnostic scoring support: baseline, experimental flag, score breakdown."""
        # Add columns to runs if not exists
        for col, ddl in [
            ("baseline_config_id", "TEXT"),
            ("baseline_metrics_json", "TEXT"),
            ("is_experimental", "INTEGER DEFAULT 0"),
            ("experimental_reason", "TEXT"),
            ("benchmark_params_json", "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {ddl}")
            except Exception:
                pass  # column already exists

        # Add columns to configurations
        for col, ddl in [
            ("score_breakdown_json", "TEXT"),
            ("estimated_ttft_ms", "REAL"),
        ]:
            try:
                conn.execute(f"ALTER TABLE configurations ADD COLUMN {col} {ddl}")
            except Exception:
                pass

        # Add checks fields to quality_results
        for col, ddl in [
            ("checks_passed", "INTEGER"),
            ("checks_total", "INTEGER"),
        ]:
            try:
                conn.execute(f"ALTER TABLE quality_results ADD COLUMN {col} {ddl}")
            except Exception:
                pass

        # Also update benchmarks ttft to estimated_ttft for clarity (keep old column)
        try:
            conn.execute("ALTER TABLE benchmarks ADD COLUMN estimated_ttft_ms REAL")
        except Exception:
            pass

    def _migration_007_benchmark_params(self, conn: sqlite3.Connection) -> None:
        """Ensure benchmark params are stored - no-op if migration 006 covered."""
        pass


# Global database manager
db_manager = DatabaseManager()
