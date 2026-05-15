"""
SQLite-backed persistence for researcher workflow state.

Stores:
- Hypotheses (user-managed research questions)
- Experiment runs (job metadata linked to hypotheses)
- Research sessions (saved v2 investigations and evidence snapshots)
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "results" / "research_state.db"
DB_PATH = Path(os.environ.get("RESEARCH_DB_PATH", str(DEFAULT_DB_PATH))).expanduser()


class ResearchStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        self._ensure_default_hypotheses()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hypotheses (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    disease_focus TEXT,
                    mechanism_type TEXT,
                    notes TEXT,
                    status TEXT NOT NULL DEFAULT 'draft',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_tested_run_id TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    job_id TEXT PRIMARY KEY,
                    hypothesis_id TEXT,
                    relation TEXT,
                    config_json TEXT NOT NULL,
                    metadata_json TEXT,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    exit_code INTEGER,
                    error TEXT,
                    run_timestamp TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_hypothesis ON experiments(hypothesis_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at DESC)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_sessions (
                    id TEXT PRIMARY KEY,
                    hypothesis_id TEXT,
                    title TEXT NOT NULL,
                    reviewer_name TEXT,
                    reviewer_email TEXT,
                    selected_entity_json TEXT NOT NULL,
                    selected_candidate_json TEXT NOT NULL,
                    run_mode TEXT NOT NULL,
                    score_threshold TEXT,
                    mechanism_weight TEXT,
                    decision TEXT NOT NULL,
                    notes TEXT,
                    evidence_state_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    exported_at REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_sessions_hypothesis ON research_sessions(hypothesis_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_sessions_updated ON research_sessions(updated_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_sessions_reviewer ON research_sessions(reviewer_email)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    user_id TEXT,
                    user_email TEXT,
                    role TEXT,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT,
                    status TEXT NOT NULL,
                    metadata_json TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_events_resource ON audit_events(resource_type, resource_id)"
            )
            conn.commit()

    def _ensure_default_hypotheses(self) -> None:
        defaults = [
            (
                "H-001",
                "Direct therapeutic mechanism",
                "Prioritize compounds likely to treat the disease through direct pathway overlap in Hetionet.",
                "direct_pathway",
            ),
            (
                "H-002",
                "Polypharmacology / multi-hop mechanism",
                "Prioritize compounds with indirect multi-hop support across disease-relevant entities.",
                "multi_hop",
            ),
            (
                "H-003",
                "Embedding-neighborhood transfer",
                "Prioritize compounds close to known treatments in embedding space for transfer-style discovery.",
                "embedding_transfer",
            ),
        ]
        with self._lock, self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) AS c FROM hypotheses").fetchone()["c"]
            if count > 0:
                return
            ts = time.time()
            for hid, name, description, mechanism in defaults:
                conn.execute(
                    """
                    INSERT INTO hypotheses
                    (id, name, description, disease_focus, mechanism_type, notes, status, created_at, updated_at, last_tested_run_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        hid,
                        name,
                        description,
                        None,
                        mechanism,
                        "",
                        "active",
                        ts,
                        ts,
                        None,
                    ),
                )
            conn.commit()

    @staticmethod
    def _row_to_hypothesis(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "disease_focus": row["disease_focus"],
            "mechanism_type": row["mechanism_type"],
            "notes": row["notes"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_tested_run_id": row["last_tested_run_id"],
        }

    @staticmethod
    def _row_to_experiment(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "hypothesis_id": row["hypothesis_id"],
            "relation": row["relation"],
            "config": json.loads(row["config_json"]) if row["config_json"] else {},
            "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            "status": row["status"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "exit_code": row["exit_code"],
            "error": row["error"],
            "run_timestamp": row["run_timestamp"],
        }

    @staticmethod
    def _json_loads(value: Optional[str], fallback: Any) -> Any:
        if not value:
            return fallback
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback

    @classmethod
    def _row_to_research_session(cls, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "hypothesis_id": row["hypothesis_id"],
            "title": row["title"],
            "reviewer_name": row["reviewer_name"],
            "reviewer_email": row["reviewer_email"],
            "selected_entity": cls._json_loads(row["selected_entity_json"], {}),
            "selected_candidate": cls._json_loads(row["selected_candidate_json"], {}),
            "run_mode": row["run_mode"],
            "score_threshold": row["score_threshold"],
            "mechanism_weight": row["mechanism_weight"],
            "decision": row["decision"],
            "notes": row["notes"],
            "evidence_state": cls._json_loads(row["evidence_state_json"], {}),
            "provenance": cls._json_loads(row["provenance_json"], []),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "exported_at": row["exported_at"],
        }

    def list_hypotheses(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hypotheses ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [self._row_to_hypothesis(r) for r in rows]

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,)
            ).fetchone()
        return self._row_to_hypothesis(row) if row else None

    def create_hypothesis(
        self,
        name: str,
        description: str,
        disease_focus: Optional[str] = None,
        mechanism_type: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "draft",
    ) -> Dict[str, Any]:
        hypothesis_id = f"H-{uuid.uuid4().hex[:6].upper()}"
        ts = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO hypotheses
                (id, name, description, disease_focus, mechanism_type, notes, status, created_at, updated_at, last_tested_run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hypothesis_id,
                    name,
                    description,
                    disease_focus,
                    mechanism_type,
                    notes or "",
                    status,
                    ts,
                    ts,
                    None,
                ),
            )
            conn.commit()
        return self.get_hypothesis(hypothesis_id)  # type: ignore[return-value]

    def update_hypothesis(self, hypothesis_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        allowed = {"name", "description", "disease_focus", "mechanism_type", "notes", "status", "last_tested_run_id"}
        fields = {k: v for k, v in patch.items() if k in allowed}
        if not fields:
            return self.get_hypothesis(hypothesis_id)
        fields["updated_at"] = time.time()
        clauses = ", ".join(f"{k} = ?" for k in fields.keys())
        vals = list(fields.values()) + [hypothesis_id]
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                f"UPDATE hypotheses SET {clauses} WHERE id = ?",
                vals,
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get_hypothesis(hypothesis_id)

    def record_experiment_created(
        self,
        job_id: str,
        hypothesis_id: Optional[str],
        relation: Optional[str],
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        ts = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments
                (job_id, hypothesis_id, relation, config_json, metadata_json, status, created_at, started_at, finished_at, exit_code, error, run_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    hypothesis_id,
                    relation,
                    json.dumps(config),
                    json.dumps(metadata or {}),
                    "queued",
                    ts,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            )
            conn.commit()

    def sync_experiment_from_job(self, job: Dict[str, Any]) -> None:
        job_id = job["id"]
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM experiments WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not existing:
                conn.execute(
                    """
                    INSERT INTO experiments
                    (job_id, hypothesis_id, relation, config_json, metadata_json, status, created_at, started_at, finished_at, exit_code, error, run_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        job.get("hypothesis_id"),
                        (job.get("flags") or {}).get("relation"),
                        json.dumps(job.get("flags") or {}),
                        json.dumps(job.get("experiment_metadata") or {}),
                        job.get("status") or "queued",
                        job.get("created_at") or time.time(),
                        job.get("started_at"),
                        job.get("finished_at"),
                        job.get("exit_code"),
                        job.get("error"),
                        None,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE experiments
                    SET hypothesis_id = ?,
                        relation = ?,
                        config_json = ?,
                        metadata_json = ?,
                        status = ?,
                        created_at = ?,
                        started_at = ?,
                        finished_at = ?,
                        exit_code = ?,
                        error = ?
                    WHERE job_id = ?
                    """,
                    (
                        job.get("hypothesis_id"),
                        (job.get("flags") or {}).get("relation"),
                        json.dumps(job.get("flags") or {}),
                        json.dumps(job.get("experiment_metadata") or {}),
                        job.get("status") or "queued",
                        job.get("created_at") or time.time(),
                        job.get("started_at"),
                        job.get("finished_at"),
                        job.get("exit_code"),
                        job.get("error"),
                        job_id,
                    ),
                )
            if job.get("hypothesis_id") and job.get("status") in {"success", "failed"}:
                conn.execute(
                    "UPDATE hypotheses SET last_tested_run_id = ?, updated_at = ? WHERE id = ?",
                    (job_id, time.time(), job.get("hypothesis_id")),
                )
            conn.commit()

    def list_experiments(self, hypothesis_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM experiments"
        params: List[Any] = []
        if hypothesis_id:
            query += " WHERE hypothesis_id = ?"
            params.append(hypothesis_id)
        query += " ORDER BY created_at DESC"
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_experiment(r) for r in rows]

    def create_research_session(
        self,
        *,
        title: str,
        reviewer_name: Optional[str],
        reviewer_email: Optional[str],
        selected_entity: Dict[str, Any],
        selected_candidate: Dict[str, Any],
        run_mode: str,
        score_threshold: Optional[str],
        mechanism_weight: Optional[str],
        decision: str,
        notes: Optional[str],
        evidence_state: Dict[str, Any],
        provenance: List[Dict[str, Any]],
        hypothesis_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        session_id = f"RS-{uuid.uuid4().hex[:8].upper()}"
        ts = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO research_sessions
                (id, hypothesis_id, title, reviewer_name, reviewer_email,
                 selected_entity_json, selected_candidate_json, run_mode,
                 score_threshold, mechanism_weight, decision, notes,
                 evidence_state_json, provenance_json, created_at, updated_at, exported_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    hypothesis_id,
                    title,
                    reviewer_name,
                    reviewer_email,
                    json.dumps(selected_entity),
                    json.dumps(selected_candidate),
                    run_mode,
                    score_threshold,
                    mechanism_weight,
                    decision,
                    notes or "",
                    json.dumps(evidence_state),
                    json.dumps(provenance),
                    ts,
                    ts,
                    None,
                ),
            )
            conn.commit()
        return self.get_research_session(session_id)  # type: ignore[return-value]

    def get_research_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return self._row_to_research_session(row) if row else None

    def list_research_sessions(
        self,
        *,
        hypothesis_id: Optional[str] = None,
        reviewer_email: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM research_sessions"
        clauses: List[str] = []
        params: List[Any] = []
        if hypothesis_id:
            clauses.append("hypothesis_id = ?")
            params.append(hypothesis_id)
        if reviewer_email:
            clauses.append("reviewer_email = ?")
            params.append(reviewer_email)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, created_at DESC"
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_research_session(r) for r in rows]

    def update_research_session(
        self,
        session_id: str,
        patch: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        json_fields = {
            "selected_entity": "selected_entity_json",
            "selected_candidate": "selected_candidate_json",
            "evidence_state": "evidence_state_json",
            "provenance": "provenance_json",
        }
        scalar_fields = {
            "hypothesis_id",
            "title",
            "reviewer_name",
            "reviewer_email",
            "run_mode",
            "score_threshold",
            "mechanism_weight",
            "decision",
            "notes",
        }
        fields: Dict[str, Any] = {}
        for key, value in patch.items():
            if key in scalar_fields:
                fields[key] = value
            elif key in json_fields:
                fields[json_fields[key]] = json.dumps(value)
        if not fields:
            return self.get_research_session(session_id)
        fields["updated_at"] = time.time()
        clauses = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values()) + [session_id]
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                f"UPDATE research_sessions SET {clauses} WHERE id = ?",
                values,
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get_research_session(session_id)

    def export_research_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        exported_at = time.time()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE research_sessions SET exported_at = ?, updated_at = ? WHERE id = ?",
                (exported_at, exported_at, session_id),
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
        session = self.get_research_session(session_id)
        if session is None:
            return None
        return {
            "evidence_packet_version": 1,
            "session": session,
            "research_context": {
                "selected_entity": session["selected_entity"],
                "selected_candidate": session["selected_candidate"],
                "run_mode": session["run_mode"],
                "score_threshold": session["score_threshold"],
                "mechanism_weight": session["mechanism_weight"],
                "hypothesis_id": session["hypothesis_id"],
            },
            "decision": {
                "decision": session["decision"],
                "notes": session["notes"],
                "reviewer_name": session["reviewer_name"],
                "reviewer_email": session["reviewer_email"],
            },
            "evidence_state": session["evidence_state"],
            "provenance": session["provenance"],
        }

    def record_audit_event(
        self,
        *,
        action: str,
        resource_type: str,
        resource_id: Optional[str],
        status: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        role: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event_id = f"AE-{uuid.uuid4().hex[:10].upper()}"
        ts = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_events
                (id, timestamp, user_id, user_email, role, action, resource_type, resource_id, status, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    ts,
                    user_id,
                    user_email,
                    role,
                    action,
                    resource_type,
                    resource_id,
                    status,
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()
        return {
            "id": event_id,
            "timestamp": ts,
            "user_id": user_id,
            "user_email": user_email,
            "role": role,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "status": status,
            "metadata": metadata or {},
        }

    def list_audit_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_events ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "user_id": row["user_id"],
                "user_email": row["user_email"],
                "role": row["role"],
                "action": row["action"],
                "resource_type": row["resource_type"],
                "resource_id": row["resource_id"],
                "status": row["status"],
                "metadata": self._json_loads(row["metadata_json"], {}),
            }
            for row in rows
        ]


research_store = ResearchStore(DB_PATH)
