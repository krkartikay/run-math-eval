import json
from pathlib import Path
import sqlite3
from uuid import uuid4


TRACE_DB_PATH = Path(__file__).resolve().parent / "traces.sqlite3"


class TraceStore:
    def __init__(
        self,
        *,
        trace_id: str | None = None,
        eval_run_id: str | None = None,
        db_path: Path = TRACE_DB_PATH,
    ):
        self.trace_id = trace_id or f"trace_{uuid4().hex}"
        self.eval_run_id = eval_run_id
        self.db_path = db_path
        self._initialize_db()
        self._insert_trace()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    eval_run_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY (eval_run_id) REFERENCES eval_runs(id)
                );

                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    eval_run_id TEXT,
                    kind TEXT NOT NULL,
                    color TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trace_id) REFERENCES traces(id),
                    FOREIGN KEY (eval_run_id) REFERENCES eval_runs(id)
                );

                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    eval_run_id TEXT,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trace_id) REFERENCES traces(id),
                    FOREIGN KEY (eval_run_id) REFERENCES eval_runs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_traces_eval_run_id
                ON traces(eval_run_id);

                CREATE INDEX IF NOT EXISTS idx_nodes_trace_id
                ON nodes(trace_id);

                CREATE INDEX IF NOT EXISTS idx_nodes_eval_run_id_kind
                ON nodes(eval_run_id, kind);

                CREATE INDEX IF NOT EXISTS idx_edges_trace_id
                ON edges(trace_id);

                CREATE TABLE IF NOT EXISTS eval_targets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    eval_run_id TEXT,
                    task_name TEXT,
                    doc_id INTEGER,
                    prompt_text TEXT,
                    expected_answer TEXT,
                    doc_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trace_id) REFERENCES traces(id),
                    FOREIGN KEY (eval_run_id) REFERENCES eval_runs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_eval_targets_trace_id
                ON eval_targets(trace_id);

                CREATE INDEX IF NOT EXISTS idx_eval_targets_eval_run_id
                ON eval_targets(eval_run_id);
                """
            )

    def _insert_trace(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO traces (id, eval_run_id, metadata_json)
                VALUES (?, ?, ?)
                """,
                (
                    self.trace_id,
                    self.eval_run_id,
                    json.dumps({}, ensure_ascii=True),
                ),
            )

    @classmethod
    def create_eval_run(
        cls,
        *,
        eval_run_id: str | None = None,
        metadata: dict | None = None,
        db_path: Path = TRACE_DB_PATH,
    ) -> str:
        run_id = eval_run_id or f"eval_{uuid4().hex}"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    metadata_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO eval_runs (id, metadata_json)
                VALUES (?, ?)
                """,
                (
                    run_id,
                    json.dumps(metadata or {}, ensure_ascii=True),
                ),
            )
        return run_id

    def add_node(
        self,
        *,
        node_id: str,
        kind: str,
        color: str,
        content,
        metadata: dict | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO nodes (
                    id,
                    trace_id,
                    eval_run_id,
                    kind,
                    color,
                    content_json,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    self.trace_id,
                    self.eval_run_id,
                    kind,
                    color,
                    json.dumps(content, ensure_ascii=True),
                    json.dumps(metadata or {}, ensure_ascii=True),
                ),
            )

    def add_edge(
        self,
        *,
        source: str,
        target: str,
        relation: str,
        metadata: dict | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO edges (
                    trace_id,
                    eval_run_id,
                    source_node_id,
                    target_node_id,
                    relation,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.trace_id,
                    self.eval_run_id,
                    source,
                    target,
                    relation,
                    json.dumps(metadata or {}, ensure_ascii=True),
                ),
            )

    def find_node_id_by_call_id(self, call_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id
                FROM nodes
                WHERE trace_id = ?
                  AND kind = 'tool_output'
                  AND json_extract(metadata_json, '$.call_id') = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (self.trace_id, call_id),
            ).fetchone()
        return None if row is None else str(row["id"])

    def add_eval_target(
        self,
        *,
        task_name: str | None,
        doc_id: int | None,
        prompt_text: str | None,
        expected_answer: str | None,
        doc: dict | None,
        metadata: dict | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO eval_targets (
                    trace_id,
                    eval_run_id,
                    task_name,
                    doc_id,
                    prompt_text,
                    expected_answer,
                    doc_json,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.trace_id,
                    self.eval_run_id,
                    task_name,
                    doc_id,
                    prompt_text,
                    expected_answer,
                    json.dumps(doc or {}, ensure_ascii=True),
                    json.dumps(metadata or {}, ensure_ascii=True),
                ),
            )
