"""
scan_cache.py — Persistent per-file scan cache
Stores expensive computed results keyed by (path, size_bytes, mtime).
Unchanged files are never re-hashed, re-embedded, or re-scored.

Components cached per file:
  - file_hash, phash, width, height, mode, EXIF  (scanner)
  - embedding (embedder — the slowest step)
  - quality_score, quality_details               (quality scorer)
"""
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import DATA_DIR

logger  = logging.getLogger(__name__)
_DB_PATH = DATA_DIR / "scan_cache.db"


class ScanCache:
    """
    Thread-safe SQLite-backed cache.  Each connection is created per call
    (SQLite handles concurrent readers/writers with WAL mode).
    """

    def __init__(self):
        self._db = str(_DB_PATH)
        self._init_db()

    # ── Schema ────────────────────────────────────────────────────────────────
    def _init_db(self):
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("""
                CREATE TABLE IF NOT EXISTS file_cache (
                    path            TEXT    NOT NULL,
                    size_bytes      INTEGER NOT NULL,
                    mtime           REAL    NOT NULL,
                    file_hash       TEXT,
                    phash           TEXT,
                    width           INTEGER,
                    height          INTEGER,
                    img_mode        TEXT,
                    embedding       TEXT,
                    quality_score   REAL,
                    quality_details TEXT,
                    taken_at        TEXT,
                    camera          TEXT,
                    gps_lat         REAL,
                    gps_lon         REAL,
                    cached_at       TEXT    NOT NULL,
                    PRIMARY KEY (path, size_bytes, mtime)
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_path ON file_cache(path)")

    def _conn(self):
        conn = sqlite3.connect(self._db, timeout=15, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Public API ────────────────────────────────────────────────────────────
    def get(self, path: str, size_bytes: int, mtime: float) -> Optional[dict]:
        """Return cached data dict or None if not found / stale."""
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM file_cache WHERE path=? AND size_bytes=? AND mtime=?",
                (path, size_bytes, round(mtime, 3))
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("embedding"):
            d["embedding"] = json.loads(d["embedding"])
        if d.get("quality_details"):
            d["quality_details"] = json.loads(d["quality_details"])
        return d

    def put(self, path: str, size_bytes: int, mtime: float, data: dict):
        """Insert or replace a cache entry.  Partial data (None fields) is fine."""
        emb = data.get("embedding")
        qd  = data.get("quality_details")
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO file_cache
                  (path, size_bytes, mtime, file_hash, phash, width, height,
                   img_mode, embedding, quality_score, quality_details,
                   taken_at, camera, gps_lat, gps_lon, cached_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                path, size_bytes, round(mtime, 3),
                data.get("file_hash"),
                data.get("phash"),
                data.get("width"),
                data.get("height"),
                data.get("img_mode"),
                json.dumps(emb) if emb is not None else None,
                data.get("quality_score"),
                json.dumps(qd)  if qd  is not None else None,
                data.get("taken_at"),
                data.get("camera"),
                data.get("gps_lat"),
                data.get("gps_lon"),
                datetime.now().isoformat(),
            ))

    def merge(self, path: str, size_bytes: int, mtime: float, updates: dict):
        """Update only the provided fields, preserving existing cached values."""
        existing = self.get(path, size_bytes, mtime) or {}
        existing.update({k: v for k, v in updates.items() if v is not None})
        self.put(path, size_bytes, mtime, existing)

    def stats(self) -> dict:
        with self._conn() as c:
            total    = c.execute("SELECT COUNT(*) FROM file_cache").fetchone()[0]
            with_emb = c.execute("SELECT COUNT(*) FROM file_cache WHERE embedding IS NOT NULL").fetchone()[0]
            with_q   = c.execute("SELECT COUNT(*) FROM file_cache WHERE quality_score IS NOT NULL").fetchone()[0]
        return {
            "cached_files":    total,
            "with_embeddings": with_emb,
            "with_quality":    with_q,
            "db_size_mb":      round(_DB_PATH.stat().st_size / 1_048_576, 1) if _DB_PATH.exists() else 0,
        }

    def preload_folder(self, folder: str) -> dict:
        """
        Load ALL cache entries whose path starts with folder in ONE SQL query.
        Returns dict keyed by (path, size_bytes, mtime) → data dict.
        Use this at the start of a scan to avoid per-file DB round-trips.
        """
        # Normalize to a consistent prefix
        prefix = str(Path(folder).resolve())
        # Make sure it ends with a separator so we don't match sibling folders
        if not prefix.endswith(('/', '\\')):
            prefix += '\\'

        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM file_cache WHERE substr(path,1,?)=? OR substr(path,1,?)=?",
                (len(prefix), prefix, len(prefix.replace('\\', '/')), prefix.replace('\\', '/'))
            ).fetchall()

        result: dict = {}
        for row in rows:
            d = dict(row)
            if d.get("embedding"):
                try:
                    d["embedding"] = json.loads(d["embedding"])
                except Exception:
                    d["embedding"] = None
            if d.get("quality_details"):
                try:
                    d["quality_details"] = json.loads(d["quality_details"])
                except Exception:
                    d["quality_details"] = {}
            key = (d["path"], d["size_bytes"], d["mtime"])
            result[key] = d

        logger.info(f"Cache preloaded: {len(result)} entries for {folder}")
        return result

    def evict_missing(self) -> int:
        """Delete entries whose source file no longer exists. Returns count removed."""
        with self._conn() as c:
            paths = [r[0] for r in c.execute("SELECT DISTINCT path FROM file_cache")]
        gone = [p for p in paths if not Path(p).exists()]
        if gone:
            with self._conn() as c:
                c.executemany("DELETE FROM file_cache WHERE path=?", [(p,) for p in gone])
            logger.info(f"Cache: evicted {len(gone)} missing-file entries")
        return len(gone)


# Module-level singleton — import and use directly
_cache: Optional[ScanCache] = None

def get_cache() -> ScanCache:
    global _cache
    if _cache is None:
        _cache = ScanCache()
        logger.info(f"Scan cache ready: {_cache.stats()['cached_files']} entries ({_cache.stats()['db_size_mb']} MB)")
    return _cache
