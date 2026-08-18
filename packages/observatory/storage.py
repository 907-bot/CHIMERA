"""Columnar Trajectory & Event Telemetry Storage Engine using DuckDB"""

from typing import List, Dict, Any, Optional
import json
import duckdb
import pandas as pd
import numpy as np
from packages.core.models import WorldState, Particle, Vector2D, Boundary
from packages.observatory.events import SimEvent, EventType


class ObservatoryStorageEngine:
    """High-Performance DuckDB Columnar Telemetry and Trajectory Store."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = duckdb.connect(database=self.db_path)
        self._init_schema()

    def _init_schema(self):
        """Initialize DuckDB tables for particle state frames and event streams."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS particle_snapshots (
                world_id VARCHAR,
                step INTEGER,
                time DOUBLE,
                dt DOUBLE,
                seed INTEGER,
                config_hash VARCHAR,
                particle_id INTEGER,
                mass DOUBLE,
                radius DOUBLE,
                pos_x DOUBLE,
                pos_y DOUBLE,
                vel_x DOUBLE,
                vel_y DOUBLE,
                force_x DOUBLE,
                force_y DOUBLE
            );
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_world_step 
            ON particle_snapshots (world_id, step);
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sim_events (
                event_id VARCHAR PRIMARY KEY,
                world_id VARCHAR,
                step INTEGER,
                time DOUBLE,
                event_type VARCHAR,
                payload_json VARCHAR,
                timestamp DOUBLE
            );
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_world_step 
            ON sim_events (world_id, step);
        """)

    def store_world_state(self, state: WorldState):
        """Store a single WorldState snapshot into DuckDB columnar table."""
        rows = []
        for p in state.particles:
            rows.append({
                "world_id": state.world_id,
                "step": state.step,
                "time": state.time,
                "dt": state.dt,
                "seed": state.seed,
                "config_hash": state.config_hash,
                "particle_id": p.id,
                "mass": p.mass,
                "radius": p.radius,
                "pos_x": p.position.x,
                "pos_y": p.position.y,
                "vel_x": p.velocity.x,
                "vel_y": p.velocity.y,
                "force_x": p.force.x,
                "force_y": p.force.y,
            })
        
        df = pd.DataFrame(rows)
        self.conn.register("df_snapshot_temp", df)
        self.conn.execute("INSERT INTO particle_snapshots SELECT * FROM df_snapshot_temp")
        self.conn.unregister("df_snapshot_temp")

    def store_trajectory(self, history: List[WorldState]):
        """Batch store a trajectory sequence of WorldState snapshots into DuckDB."""
        rows = []
        for state in history:
            for p in state.particles:
                rows.append({
                    "world_id": state.world_id,
                    "step": state.step,
                    "time": state.time,
                    "dt": state.dt,
                    "seed": state.seed,
                    "config_hash": state.config_hash,
                    "particle_id": p.id,
                    "mass": p.mass,
                    "radius": p.radius,
                    "pos_x": p.position.x,
                    "pos_y": p.position.y,
                    "vel_x": p.velocity.x,
                    "vel_y": p.velocity.y,
                    "force_x": p.force.x,
                    "force_y": p.force.y,
                })

        if not rows:
            return

        df = pd.DataFrame(rows)
        self.conn.register("df_trajectory_temp", df)
        self.conn.execute("INSERT INTO particle_snapshots SELECT * FROM df_trajectory_temp")
        self.conn.unregister("df_trajectory_temp")

    def store_event(self, event: SimEvent):
        """Store a simulation event into DuckDB event stream table."""
        self.conn.execute(
            """
            INSERT INTO sim_events VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.world_id,
                event.step,
                event.time,
                event.event_type.value if isinstance(event.event_type, EventType) else str(event.event_type),
                json.dumps(event.payload),
                event.timestamp,
            ),
        )

    def query_trajectory_slice(
        self,
        world_id: str,
        start_step: int = 0,
        end_step: Optional[int] = None
    ) -> List[WorldState]:
        """Query trajectory slice in range [start_step, end_step] with sub-10ms latency."""
        query = """
            SELECT world_id, step, time, dt, seed, config_hash,
                   particle_id, mass, radius, pos_x, pos_y, vel_x, vel_y, force_x, force_y
            FROM particle_snapshots
            WHERE world_id = ? AND step >= ?
        """
        params = [world_id, start_step]

        if end_step is not None:
            query += " AND step <= ?"
            params.append(end_step)

        query += " ORDER BY step ASC, particle_id ASC"

        rows = self.conn.execute(query, params).fetchall()
        if not rows:
            return []

        # High-performance tuple unpacking group-by loop
        states_dict: Dict[int, Dict[str, Any]] = {}
        for r in rows:
            w_id, step_val, t, dt, seed, cfg_hash, p_id, mass, radius, px, py, vx, vy, fx, fy = r
            if step_val not in states_dict:
                states_dict[step_val] = {
                    "world_id": w_id,
                    "step": step_val,
                    "time": t,
                    "dt": dt,
                    "seed": seed,
                    "config_hash": cfg_hash,
                    "particles": [],
                }

            p = Particle.model_construct(
                id=p_id,
                mass=mass,
                radius=radius,
                position=Vector2D.model_construct(x=px, y=py),
                velocity=Vector2D.model_construct(x=vx, y=vy),
                force=Vector2D.model_construct(x=fx, y=fy),
            )
            states_dict[step_val]["particles"].append(p)

        states = []
        for step_key in sorted(states_dict.keys()):
            data = states_dict[step_key]
            state = WorldState.model_construct(
                world_id=data["world_id"],
                step=data["step"],
                time=data["time"],
                dt=data["dt"],
                particles=data["particles"],
                boundary=Boundary(),
                seed=data["seed"],
                config_hash=data["config_hash"],
            )
            states.append(state)

        return states

    def query_events(
        self,
        world_id: str,
        event_type: Optional[str] = None,
        start_step: int = 0,
        end_step: Optional[int] = None,
    ) -> List[SimEvent]:
        """Query simulation events matching filters."""
        query = "SELECT event_id, world_id, step, time, event_type, payload_json, timestamp FROM sim_events WHERE world_id = ? AND step >= ?"
        params = [world_id, start_step]

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if end_step is not None:
            query += " AND step <= ?"
            params.append(end_step)

        query += " ORDER BY step ASC"

        df = self.conn.execute(query, params).fetchdf()
        events = []
        for _, row in df.iterrows():
            ev = SimEvent(
                event_id=str(row["event_id"]),
                world_id=str(row["world_id"]),
                step=int(row["step"]),
                time=float(row["time"]),
                event_type=EventType(row["event_type"]),
                payload=json.loads(row["payload_json"]),
                timestamp=float(row["timestamp"]),
            )
            events.append(ev)
        return events

    def count_recorded_steps(self, world_id: str) -> int:
        """Count unique steps recorded for a world."""
        res = self.conn.execute(
            "SELECT COUNT(DISTINCT step) FROM particle_snapshots WHERE world_id = ?",
            [world_id],
        ).fetchone()
        return res[0] if res else 0

    def close(self):
        """Close DuckDB connection."""
        self.conn.close()
