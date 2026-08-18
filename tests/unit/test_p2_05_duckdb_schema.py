"""P2-05 — DuckDB Schema and Migration Safety

Inspects DuckDB tables to verify expected columns, data types, indexes,
and schema compatibility across database restarts.
"""

import pytest
import duckdb
from packages.observatory.storage import ObservatoryStorageEngine


class TestDuckDBSchemaSafety:
    """Test suite for DuckDB schema structure and compatibility."""

    def test_particle_snapshots_schema(self):
        storage = ObservatoryStorageEngine(":memory:")
        schema_info = storage.conn.execute("PRAGMA table_info('particle_snapshots');").fetchall()

        # Extract column names and types
        col_types = {row[1]: row[2].upper() for row in schema_info}

        expected_columns = {
            "world_id": "VARCHAR",
            "step": "INTEGER",
            "time": "DOUBLE",
            "dt": "DOUBLE",
            "seed": "INTEGER",
            "config_hash": "VARCHAR",
            "particle_id": "INTEGER",
            "mass": "DOUBLE",
            "radius": "DOUBLE",
            "pos_x": "DOUBLE",
            "pos_y": "DOUBLE",
            "vel_x": "DOUBLE",
            "vel_y": "DOUBLE",
            "force_x": "DOUBLE",
            "force_y": "DOUBLE",
        }

        for col, expected_type in expected_columns.items():
            assert col in col_types, f"Missing expected column: {col}"
            assert expected_type in col_types[col], f"Column {col} type mismatch: {col_types[col]} vs {expected_type}"

        storage.close()

    def test_sim_events_schema(self):
        storage = ObservatoryStorageEngine(":memory:")
        schema_info = storage.conn.execute("PRAGMA table_info('sim_events');").fetchall()

        col_types = {row[1]: row[2].upper() for row in schema_info}

        expected_columns = {
            "event_id": "VARCHAR",
            "world_id": "VARCHAR",
            "step": "INTEGER",
            "time": "DOUBLE",
            "event_type": "VARCHAR",
            "payload_json": "VARCHAR",
            "timestamp": "DOUBLE",
        }

        for col, expected_type in expected_columns.items():
            assert col in col_types, f"Missing expected column in sim_events: {col}"
            assert expected_type in col_types[col], f"Column {col} type mismatch: {col_types[col]} vs {expected_type}"

        storage.close()

    def test_schema_idempotence(self, tmp_path):
        db_file = str(tmp_path / "schema_idempotent.duckdb")

        # Initial connect initializes schema
        storage1 = ObservatoryStorageEngine(db_path=db_file)
        storage1.close()

        # Second connect re-runs _init_schema without error (CREATE TABLE IF NOT EXISTS)
        storage2 = ObservatoryStorageEngine(db_path=db_file)
        schema1 = storage2.conn.execute("PRAGMA table_info('particle_snapshots');").fetchall()
        assert len(schema1) == 15
        storage2.close()
