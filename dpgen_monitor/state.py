from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING, Iterable, Iterator

if TYPE_CHECKING:
    from .dpgen import IterationSnapshot


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
                    source_identity TEXT,
                    generation INTEGER NOT NULL DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 1,
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

                CREATE TABLE IF NOT EXISTS delivery_history (
                    event_key TEXT NOT NULL,
                    notifier TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (event_key, notifier, content_hash)
                );

                CREATE TABLE IF NOT EXISTS committee_replays (
                    model_iteration INTEGER NOT NULL,
                    source_iteration INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    summary_file TEXT,
                    last_error TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (model_iteration, source_iteration)
                );

                CREATE TABLE IF NOT EXISTS parameter_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    target_iteration INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    parameter_file TEXT NOT NULL,
                    parameter_sha256 TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    proposed_job_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    review_note TEXT,
                    backup_file TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    applied_at TEXT
                );
                """
            )
            stage_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(iteration_stages)")
            }
            if "source_identity" not in stage_columns:
                connection.execute(
                    "ALTER TABLE iteration_stages ADD COLUMN source_identity TEXT"
                )
            if "generation" not in stage_columns:
                connection.execute(
                    "ALTER TABLE iteration_stages "
                    "ADD COLUMN generation INTEGER NOT NULL DEFAULT 0"
                )
            if "active" not in stage_columns:
                connection.execute(
                    "ALTER TABLE iteration_stages "
                    "ADD COLUMN active INTEGER NOT NULL DEFAULT 1"
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
                    active=1,
                    updated_at=excluded.updated_at
                """,
                (iteration, task, _utc_now()),
            )

    @staticmethod
    def _delete_iteration_results(
        connection: sqlite3.Connection, iteration: int
    ) -> None:
        connection.execute("DELETE FROM statistics WHERE iteration=?", (iteration,))
        connection.execute("DELETE FROM evaluations WHERE iteration=?", (iteration,))
        connection.execute(
            "DELETE FROM committee_replays "
            "WHERE model_iteration=? OR source_iteration=?",
            (iteration, iteration),
        )
        connection.execute(
            "UPDATE parameter_proposals SET status='stale', updated_at=? "
            "WHERE target_iteration>=? AND status IN ('pending', 'approved')",
            (_utc_now(), iteration),
        )

    def reconcile_iterations(
        self, snapshots: Iterable["IterationSnapshot"]
    ) -> tuple[dict[int, int], list[str]]:
        """Bind state to current iteration directories and detect rollbacks.

        A DP-GEN recovery can delete later iterations or reuse an iteration
        number after resetting record.dpgen. In that case cached calculation
        results for the superseded generation must not be reused. Delivery
        history is retained so identical regenerated content is not sent again.
        """
        current = {snapshot.iteration: snapshot for snapshot in snapshots}
        generations: dict[int, int] = {}
        changes: list[str] = []
        now = _utc_now()

        with self.transaction() as connection:
            stored = {
                int(row["iteration"]): row
                for row in connection.execute("SELECT * FROM iteration_stages")
            }

            if current:
                highest = max(current)
                for iteration, row in stored.items():
                    if (
                        iteration > highest
                        and int(row["active"] or 0) == 1
                        and iteration not in current
                    ):
                        self._delete_iteration_results(connection, iteration)
                        generation = int(row["generation"] or 0) + 1
                        connection.execute(
                            """
                            UPDATE iteration_stages
                            SET task=NULL, source_identity=NULL, generation=?,
                                active=0, updated_at=?
                            WHERE iteration=?
                            """,
                            (generation, now, iteration),
                        )
                        changes.append(
                            f"iter.{iteration:06d} 已从运行目录移除"
                        )

            for iteration, snapshot in current.items():
                row = stored.get(iteration)
                generation = int(row["generation"] or 0) if row else 0
                reason: str | None = None
                if row and int(row["active"] or 0) == 1:
                    old_identity = row["source_identity"]
                    old_task = row["task"]
                    if old_identity and old_identity != snapshot.source_identity:
                        reason = "迭代目录已重建"
                    elif (
                        old_task is not None
                        and snapshot.task is not None
                        and int(snapshot.task) < int(old_task)
                    ):
                        reason = f"阶段从 task {int(old_task):02d} 回退到 task {snapshot.task:02d}"

                if reason:
                    self._delete_iteration_results(connection, iteration)
                    generation += 1
                    changes.append(f"iter.{iteration:06d} {reason}")

                connection.execute(
                    """
                    INSERT INTO iteration_stages(
                        iteration, task, source_identity, generation, active, updated_at
                    ) VALUES (?, ?, ?, ?, 1, ?)
                    ON CONFLICT(iteration) DO UPDATE SET
                        task=excluded.task,
                        source_identity=excluded.source_identity,
                        generation=excluded.generation,
                        active=1,
                        updated_at=excluded.updated_at
                    """,
                    (
                        iteration,
                        snapshot.task,
                        snapshot.source_identity,
                        generation,
                        now,
                    ),
                )
                generations[iteration] = generation

        return generations, changes

    def get_iteration_generation(self, iteration: int) -> int:
        row = self.connection.execute(
            "SELECT generation FROM iteration_stages WHERE iteration=? AND active=1",
            (iteration,),
        ).fetchone()
        return int(row["generation"] or 0) if row else 0

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
            """
            SELECT statistics.* FROM statistics
            JOIN iteration_stages USING(iteration)
            WHERE iteration_stages.active=1
            ORDER BY statistics.iteration
            """
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

    def evaluations_complete(
        self, iteration: int, phase: str, model_ids: Iterable[str]
    ) -> bool:
        expected = tuple(model_ids)
        if not expected:
            return False
        placeholders = ", ".join("?" for _ in expected)
        row = self.connection.execute(
            f"""
            SELECT COUNT(*) AS complete_count
            FROM evaluations
            WHERE iteration=? AND phase=? AND status='complete'
              AND model_id IN ({placeholders})
            """,
            (iteration, phase, *expected),
        ).fetchone()
        return int(row["complete_count"]) == len(expected)

    def set_committee_replay(
        self,
        model_iteration: int,
        source_iteration: int,
        status: str,
        summary_file: str | None = None,
        last_error: str | None = None,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO committee_replays(
                    model_iteration, source_iteration, status,
                    summary_file, last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_iteration, source_iteration) DO UPDATE SET
                    status=excluded.status,
                    summary_file=excluded.summary_file,
                    last_error=excluded.last_error,
                    updated_at=excluded.updated_at
                """,
                (
                    model_iteration,
                    source_iteration,
                    status,
                    summary_file,
                    last_error,
                    _utc_now(),
                ),
            )

    def get_committee_replay(
        self, model_iteration: int, source_iteration: int
    ) -> dict | None:
        row = self.connection.execute(
            """
            SELECT * FROM committee_replays
            WHERE model_iteration=? AND source_iteration=?
            """,
            (model_iteration, source_iteration),
        ).fetchone()
        return dict(row) if row else None

    def list_committee_replays(self) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT committee_replays.* FROM committee_replays
            JOIN iteration_stages AS model_stage
              ON model_stage.iteration=committee_replays.model_iteration
            JOIN iteration_stages AS source_stage
              ON source_stage.iteration=committee_replays.source_iteration
            WHERE model_stage.active=1 AND source_stage.active=1
            ORDER BY model_iteration, source_iteration
            """
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _decode_parameter_proposal(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        result = dict(row)
        result["proposed_job"] = json.loads(result.pop("proposed_job_json"))
        result["evidence"] = json.loads(result.pop("evidence_json"))
        return result

    def create_parameter_proposal(
        self,
        *,
        proposal_id: str,
        target_iteration: int,
        status: str,
        parameter_file: str,
        parameter_sha256: str,
        strategy: str,
        proposed_job: dict,
        evidence: list[dict],
    ) -> None:
        now = _utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE parameter_proposals
                SET status='stale', updated_at=?
                WHERE target_iteration=? AND proposal_id<>?
                  AND status IN ('pending', 'approved')
                """,
                (now, target_iteration, proposal_id),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO parameter_proposals(
                    proposal_id, target_iteration, status, parameter_file,
                    parameter_sha256, strategy, proposed_job_json,
                    evidence_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal_id,
                    target_iteration,
                    status,
                    parameter_file,
                    parameter_sha256,
                    strategy,
                    json.dumps(proposed_job, ensure_ascii=False, sort_keys=True),
                    json.dumps(evidence, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )

    def get_parameter_proposal(self, proposal_id: str) -> dict | None:
        row = self.connection.execute(
            "SELECT * FROM parameter_proposals WHERE proposal_id=?",
            (proposal_id,),
        ).fetchone()
        return self._decode_parameter_proposal(row)

    def list_parameter_proposals(self, status: str | None = None) -> list[dict]:
        if status is None:
            rows = self.connection.execute(
                "SELECT * FROM parameter_proposals ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM parameter_proposals WHERE status=? "
                "ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        return [self._decode_parameter_proposal(row) for row in rows]

    def transition_parameter_proposal(
        self,
        proposal_id: str,
        *,
        expected_statuses: tuple[str, ...],
        status: str,
        review_note: str | None = None,
        backup_file: str | None = None,
    ) -> dict:
        if not expected_statuses:
            raise ValueError("参数建议状态转换缺少来源状态")
        now = _utc_now()
        placeholders = ", ".join("?" for _ in expected_statuses)
        reviewed_at = now if status in {"approved", "rejected"} else None
        applied_at = now if status == "applied" else None
        with self.transaction() as connection:
            cursor = connection.execute(
                f"""
                UPDATE parameter_proposals
                SET status=?, review_note=COALESCE(?, review_note),
                    backup_file=COALESCE(?, backup_file), updated_at=?,
                    reviewed_at=COALESCE(?, reviewed_at),
                    applied_at=COALESCE(?, applied_at)
                WHERE proposal_id=? AND status IN ({placeholders})
                """,
                (
                    status,
                    review_note,
                    backup_file,
                    now,
                    reviewed_at,
                    applied_at,
                    proposal_id,
                    *expected_statuses,
                ),
            )
            if cursor.rowcount != 1:
                row = connection.execute(
                    "SELECT status FROM parameter_proposals WHERE proposal_id=?",
                    (proposal_id,),
                ).fetchone()
                current = row["status"] if row else "不存在"
                raise ValueError(
                    f"参数建议状态为 {current}，不能转换为 {status}"
                )
        result = self.get_parameter_proposal(proposal_id)
        assert result is not None
        return result

    def is_delivered(self, event_key: str, notifier: str) -> bool:
        row = self.connection.execute(
            "SELECT status FROM deliveries WHERE event_key=? AND notifier=?",
            (event_key, notifier),
        ).fetchone()
        return bool(row and row["status"] == "delivered")

    def has_delivered_content(
        self, event_key: str, notifier: str, content_hash: str
    ) -> bool:
        """Check immutable content history, adopting legacy delivery rows once.

        Older databases know that an event was delivered but do not contain a
        content digest. The first post-migration observation binds that legacy
        delivery to the current content instead of sending it again.
        """
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT status FROM delivery_history
                WHERE event_key=? AND notifier=? AND content_hash=?
                """,
                (event_key, notifier, content_hash),
            ).fetchone()
            if row:
                return row["status"] == "delivered"

            history_exists = connection.execute(
                """
                SELECT 1 FROM delivery_history
                WHERE event_key=? AND notifier=? LIMIT 1
                """,
                (event_key, notifier),
            ).fetchone()
            if history_exists:
                return False

            legacy = connection.execute(
                """
                SELECT status, attempts, last_error, updated_at
                FROM deliveries WHERE event_key=? AND notifier=?
                """,
                (event_key, notifier),
            ).fetchone()
            if not legacy or legacy["status"] != "delivered":
                return False
            connection.execute(
                """
                INSERT INTO delivery_history(
                    event_key, notifier, content_hash, status,
                    attempts, last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_key,
                    notifier,
                    content_hash,
                    legacy["status"],
                    legacy["attempts"],
                    legacy["last_error"],
                    legacy["updated_at"],
                ),
            )
            return True

    def record_delivery(
        self,
        event_key: str,
        notifier: str,
        success: bool,
        error: str | None = None,
        content_hash: str | None = None,
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
            if content_hash is not None:
                connection.execute(
                    """
                    INSERT INTO delivery_history(
                        event_key, notifier, content_hash, status,
                        attempts, last_error, updated_at
                    ) VALUES (?, ?, ?, ?, 1, ?, ?)
                    ON CONFLICT(event_key, notifier, content_hash) DO UPDATE SET
                        status=excluded.status,
                        attempts=delivery_history.attempts + 1,
                        last_error=excluded.last_error,
                        updated_at=excluded.updated_at
                    """,
                    (
                        event_key,
                        notifier,
                        content_hash,
                        status,
                        error,
                        _utc_now(),
                    ),
                )

    def list_deliveries(self, limit: int = 100) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT event_key, notifier, status, attempts, last_error, updated_at
            FROM deliveries
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
        return [dict(row) for row in rows]

    def status_summary(self) -> dict[str, int]:
        queries = {
            "iterations": "SELECT COUNT(*) FROM iteration_stages WHERE active=1",
            "statistics": (
                "SELECT COUNT(*) FROM statistics "
                "JOIN iteration_stages USING(iteration) "
                "WHERE iteration_stages.active=1"
            ),
            "evaluations_complete": (
                "SELECT COUNT(*) FROM evaluations "
                "JOIN iteration_stages USING(iteration) "
                "WHERE status='complete' AND iteration_stages.active=1"
            ),
            "committee_replays_complete": (
                "SELECT COUNT(*) FROM committee_replays "
                "JOIN iteration_stages AS model_stage "
                "ON model_stage.iteration=committee_replays.model_iteration "
                "JOIN iteration_stages AS source_stage "
                "ON source_stage.iteration=committee_replays.source_iteration "
                "WHERE committee_replays.status='complete' "
                "AND model_stage.active=1 AND source_stage.active=1"
            ),
            "parameter_proposals_pending": (
                "SELECT COUNT(*) FROM parameter_proposals WHERE status='pending'"
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
