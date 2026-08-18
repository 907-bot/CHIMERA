"""Append-Only SQLite Hypothesis Registry for CHIMERA Phase 3.

Implements Rule 6 (Immutable Scientific Evidence):
  Failed hypotheses and counter-evidence MUST be retained to prevent
  circular reasoning. The registry is append-only — hypotheses can be
  created and their status updated, but NEVER deleted.

Storage backend: SQLite with WAL mode for concurrent reads.
"""

from __future__ import annotations
import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import List, Optional, Literal
from packages.symbolic.hypothesis import Hypothesis, HypothesisParameters, PredictionMetrics


_DEFAULT_DB_PATH = "experiments/hypothesis_registry.db"


class HypothesisRegistry:
    """Append-only SQLite registry for scientific hypotheses.

    Schema:
        hypotheses(
            id TEXT PRIMARY KEY,
            world_name TEXT NOT NULL,
            solver TEXT NOT NULL,
            candidate_equation TEXT NOT NULL,
            parameters_json TEXT NOT NULL,
            metrics_json TEXT,
            evidence_step_range TEXT NOT NULL,
            status TEXT NOT NULL,
            falsification_evidence TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )

    Write semantics:
        - INSERT: register_hypothesis()
        - UPDATE status only: update_status()   (no row deletions ever)
        - SELECT: all query methods

    Args:
        db_path: Path to the SQLite database file. Use ':memory:' for tests.
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self.db_path = db_path

        # Ensure directory exists if writing to file
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._create_schema()

    def _create_schema(self) -> None:
        """Create hypothesis table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                id                    TEXT PRIMARY KEY,
                world_name            TEXT NOT NULL,
                solver                TEXT NOT NULL,
                candidate_equation    TEXT NOT NULL,
                parameters_json       TEXT NOT NULL,
                metrics_json          TEXT,
                evidence_step_range   TEXT NOT NULL,
                status                TEXT NOT NULL,
                falsification_evidence TEXT,
                created_at            TEXT NOT NULL,
                updated_at            TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def register_hypothesis(self, hypothesis: Hypothesis) -> str:
        """Append a new hypothesis record to the registry.

        Args:
            hypothesis: A Hypothesis in CANDIDATE state.

        Returns:
            The hypothesis UUID string.

        Raises:
            ValueError: If hypothesis with same ID already exists.
        """
        now = datetime.now(timezone.utc).isoformat()
        params_json = json.dumps({
            "values": hypothesis.parameters.values,
            "uncertainties": hypothesis.parameters.uncertainties,
        })
        metrics_json = None
        if hypothesis.metrics:
            metrics_json = json.dumps(hypothesis.metrics.model_dump())

        self.conn.execute(
            """
            INSERT INTO hypotheses
                (id, world_name, solver, candidate_equation, parameters_json,
                 metrics_json, evidence_step_range, status, falsification_evidence,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hypothesis.id,
                hypothesis.world_name,
                hypothesis.solver,
                hypothesis.candidate_equation,
                params_json,
                metrics_json,
                json.dumps(list(hypothesis.evidence_step_range)),
                hypothesis.status,
                hypothesis.falsification_evidence,
                hypothesis.created_at,
                now,
            ),
        )
        self.conn.commit()
        return hypothesis.id

    def update_status(
        self,
        hypothesis_id: str,
        new_status: Literal["VALIDATED", "FALSIFIED"],
        metrics: Optional[PredictionMetrics] = None,
        falsification_evidence: Optional[str] = None,
    ) -> None:
        """Update hypothesis status (VALIDATED or FALSIFIED). Rows are never deleted.

        Args:
            hypothesis_id:         UUID of the hypothesis to update.
            new_status:            New lifecycle status.
            metrics:               Updated prediction metrics (optional).
            falsification_evidence: Evidence string if FALSIFIED (optional).
        """
        if new_status not in ("VALIDATED", "FALSIFIED"):
            raise ValueError(f"Invalid status transition to '{new_status}'. Allowed: 'VALIDATED', 'FALSIFIED'")

        now = datetime.now(timezone.utc).isoformat()
        metrics_json = None
        if metrics:
            metrics_json = json.dumps(metrics.model_dump())

        cursor = self.conn.execute(
            """
            UPDATE hypotheses
            SET status = ?, metrics_json = ?, falsification_evidence = ?, updated_at = ?
            WHERE id = ?
            """,
            (new_status, metrics_json, falsification_evidence, now, hypothesis_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"Hypothesis with id '{hypothesis_id}' not found in registry")
        self.conn.commit()

    def get_by_id(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Retrieve a single hypothesis by UUID.

        Args:
            hypothesis_id: UUID string.

        Returns:
            Hypothesis object or None if not found.
        """
        row = self.conn.execute(
            "SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,)
        ).fetchone()
        return self._row_to_hypothesis(row) if row else None

    def get_by_world(
        self,
        world_name: str,
        status_filter: Optional[str] = None,
    ) -> List[Hypothesis]:
        """Retrieve all hypotheses for a given world, optionally filtered by status.

        Args:
            world_name:    Name of the benchmark world.
            status_filter: 'CANDIDATE', 'VALIDATED', or 'FALSIFIED' (None = all).

        Returns:
            List of Hypothesis objects ordered by created_at ascending.
        """
        if status_filter:
            rows = self.conn.execute(
                "SELECT * FROM hypotheses WHERE world_name = ? AND status = ? ORDER BY created_at ASC",
                (world_name, status_filter),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM hypotheses WHERE world_name = ? ORDER BY created_at ASC",
                (world_name,),
            ).fetchall()

        return [h for row in rows if (h := self._row_to_hypothesis(row)) is not None]

    def count_all(self) -> int:
        """Return total number of hypothesis records (including falsified — immutable)."""
        return self.conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]

    def _row_to_hypothesis(self, row: tuple) -> Optional[Hypothesis]:
        """Deserialise a DB row tuple into a Hypothesis object.

        Row column order:
            0  id
            1  world_name
            2  solver
            3  candidate_equation
            4  parameters_json
            5  metrics_json
            6  evidence_step_range
            7  status
            8  falsification_evidence
            9  created_at
            10 updated_at
        """
        if row is None:
            return None

        params_data = json.loads(row[4])
        parameters = HypothesisParameters(
            values=params_data["values"],
            uncertainties=params_data.get("uncertainties", {}),
        )

        metrics = None
        if row[5]:
            m = json.loads(row[5])
            metrics = PredictionMetrics(**m)

        step_range = json.loads(row[6])

        return Hypothesis(
            id=row[0],
            world_name=row[1],
            solver=row[2],
            candidate_equation=row[3],
            parameters=parameters,
            metrics=metrics,
            evidence_step_range=tuple(step_range),
            status=row[7],
            falsification_evidence=row[8],
            created_at=row[9],
        )

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
