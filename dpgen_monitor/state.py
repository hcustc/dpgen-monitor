from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Iterator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    """SQLite-backed durable state for restart-safe monitoring."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def _create_schema(self) -> None:
        with self.transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS iteration_stages (
                    iteration INTEGER PRIMARY KEY,
                    task INTEGER,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS statistics (
                    iteration INTEGER PRIMARY KEY,
                    candidate_count INTEGER NOT NULL,
                    candidate_total INTEGER NOT NULL,
                    candidate_percent REAL NOT NULL,
                    failed_count INTEGER NOT NULL,
                    failed_total INTEGER NOT NULL,
                    failed_percent REAL NOT NULL,
                    accurate_count INTEGER NOT NULL,
                    accurate_total INTEGER NOT NULL,
                    accurate_percent REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS deliveries (
                    event_key TEXT NOT NULL,
                    notifier TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (event_key, notifier)
                );
                """
            )
            evaluation_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(evaluations)")
            }
            if evaluation_columns and "phase" not in evaluation_columns:
                connection.execute("ALTER TABLE evaluations RENAME TO evaluations_legacy")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluations (
                    iteration INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    force_file TEXT,
                    last_error TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (iteration, phase, model_id)
                )
                """
            )
            if evaluation_columns and "phase" not in evaluation_columns:
                connection.execute(
                    """
                    INSERT INTO evaluations(
                        iteration, phase, model_id, status,
                        force_file, last_error, updated_at
                    )
                    SELECT iteration, 'absorption', model_id, status,
                           force_file, last_error, updated_at
                    FROM evaluations_legacy
                    """
                )
                connection.execute("DROP TABLE evaluations_legacy")

    def upsert_stage(self, iteration: int, task: int | None) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO iteration_stages(iteration, task, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(iteration) DO UPDATE SET
                    task=excluded.task,
                    updated_at=excluded.updated_at
                """,
                (iteration, task, _utc_now()),
            )

    def upsert_statistics(self, iteration: int, stats: dict[str, int | float]) -> None:
        columns = (
            "candidate_count", "candidate_total", "candidate_percent",
            "failed_count", "failed_total", "failed_percent",
            "accurate_count", "accurate_total", "accurate_percent",
        )
        values = [stats.get(column, 0) for column in columns]
        with self.transaction() as connection:
            connection.execute(
                f"""
                INSERT INTO statistics(iteration, {', '.join(columns)}, updated_at)
                VALUES (?, {', '.join('?' for _ in columns)}, ?)
                ON CONFLICT(iteration) DO UPDATE SET
                    {', '.join(f'{column}=excluded.{column}' for column in columns)},
                    updated_at=excluded.updated_at
                """,
                (iteration, *values, _utc_now()),
            )

    def list_statistics(self) -> list[dict[str, int | float]]:
        rows = self.connection.execute(
            "SELECT * FROM statistics ORDER BY iteration"
        ).fetchall()
        return [dict(row) for row in rows]

    def set_evaluation(
        self,
        iteration: int,
        phase: str,
        model_id: str,
        status: str,
        force_file: str | None = None,
        last_error: str | None = None,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO evaluations(
                    iteration, phase, model_id, status,
                    force_file, last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(iteration, phase, model_id) DO UPDATE SET
                    status=excluded.status,
                    force_file=excluded.force_file,
                    last_error=excluded.last_error,
                    updated_at=excluded.updated_at
                """,
                (
                    iteration, phase, model_id, status,
                    force_file, last_error, _utc_now(),
                ),
            )

    def get_evaluation(
        self, iteration: int, phase: str, model_id: str
    ) -> dict | None:
        row = self.connection.execute(
            """
            SELECT * FROM evaluations
            WHERE iteration=? AND phase=? AND model_id=?
            """,
            (iteration, phase, model_id),
        ).fetchone()
        return dict(row) if row else None

    def is_delivered(self, event_key: str, notifier: str) -> bool:
        row = self.connection.execute(
            "SELECT status FROM deliveries WHERE event_key=? AND notifier=?",
            (event_key, notifier),
        ).fetchone()
        return bool(row and row["status"] == "delivered")

    def record_delivery(
        self,
        event_key: str,
        notifier: str,
        success: bool,
        error: str | None = None,
    ) -> None:
        status = "delivered" if success else "failed"
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO deliveries(
                    event_key, notifier, status, attempts, last_error, updated_at
                ) VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT(event_key, notifier) DO UPDATE SET
                    status=excluded.status,
                    attempts=deliveries.attempts + 1,
                    last_error=excluded.last_error,
                    updated_at=excluded.updated_at
                """,
                (event_key, notifier, status, error, _utc_now()),
            )

    def status_summary(self) -> dict[str, int]:
        queries = {
            "iterations": "SELECT COUNT(*) FROM iteration_stages",
            "statistics": "SELECT COUNT(*) FROM statistics",
            "evaluations_complete": (
                "SELECT COUNT(*) FROM evaluations WHERE status='complete'"
            ),
            "deliveries_complete": (
                "SELECT COUNT(*) FROM deliveries WHERE status='delivered'"
            ),
            "deliveries_failed": "SELECT COUNT(*) FROM deliveries WHERE status='failed'",
        }
        return {
            key: int(self.connection.execute(query).fetchone()[0])
            for key, query in queries.items()
        }
