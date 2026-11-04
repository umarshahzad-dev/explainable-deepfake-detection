from contextlib import closing
import json
import sqlite3
from deepfake_detection.common.paths import Workspace


class HistoryStore:
    def __init__(self, workspace=None):
        self.workspace = workspace or Workspace()
        self.path = self.workspace.output(self.workspace.runs / "service" / "history.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id INTEGER PRIMARY KEY, run_id TEXT NOT NULL,
                    weights_sha256 TEXT NOT NULL, preprocessing_hash TEXT NOT NULL,
                    UNIQUE(weights_sha256, preprocessing_hash));
                CREATE TABLE IF NOT EXISTS analyses (
                    analysis_id INTEGER PRIMARY KEY, model_id INTEGER NOT NULL REFERENCES models(model_id),
                    media_sha256 TEXT NOT NULL, source_type TEXT NOT NULL, status TEXT NOT NULL,
                    face_count INTEGER NOT NULL CHECK(face_count >= 0), elapsed_seconds REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, result_json TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS analyses_model ON analyses(model_id);
            """)
            connection.commit()

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def record(self, media_hash, source_type, result):
        encoded = json.dumps(result, allow_nan=False)
        with closing(self.connect()) as connection, connection:
            connection.execute("INSERT OR IGNORE INTO models(run_id, weights_sha256, preprocessing_hash) VALUES (?,?,?)",
                               (result["run_id"], result["weights_sha256"], result["preprocessing_hash"]))
            model_id = connection.execute("SELECT model_id FROM models WHERE weights_sha256=? AND preprocessing_hash=?",
                                          (result["weights_sha256"], result["preprocessing_hash"])).fetchone()[0]
            cursor = connection.execute("INSERT INTO analyses(model_id, media_sha256, source_type, status, face_count, elapsed_seconds, result_json) VALUES (?,?,?,?,?,?,?)",
                                        (model_id, media_hash, source_type, result["status"], len(result["faces"]), result["elapsed_seconds"], encoded))
            return cursor.lastrowid

    def recent(self, limit=20):
        limit = max(1, min(int(limit), 100))
        with closing(self.connect()) as connection:
            rows = connection.execute("SELECT a.analysis_id,a.created_at,a.source_type,a.status,a.face_count,a.elapsed_seconds,m.run_id,m.weights_sha256 FROM analyses a JOIN models m ON m.model_id=a.model_id ORDER BY a.analysis_id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(row) for row in rows]
