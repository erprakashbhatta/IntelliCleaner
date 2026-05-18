"""
memory/rag_store.py — RAG Memory Store
Persists every human decision as a structured memory record.
On new groups, retrieves the K most similar past decisions as few-shot
context for the LLM, enabling continuous learning without retraining.
"""
import json
import uuid
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH, CHROMA_DIR, RAG_TOP_K_MEMORIES

logger = logging.getLogger(__name__)


# ─── SQLite schema ────────────────────────────────────────────────────────────
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS decisions (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    group_id        TEXT NOT NULL,
    tier            TEXT,
    kept_path       TEXT,
    deleted_paths   TEXT,         -- JSON list
    reason_tags     TEXT,         -- JSON list
    human_notes     TEXT,
    llm_suggestion  TEXT,         -- path LLM suggested
    llm_confidence  REAL,
    human_accepted  INTEGER,      -- 1 = accepted, 0 = rejected/overrode
    quality_scores  TEXT,         -- JSON dict: path → score
    group_embedding TEXT,         -- JSON list (mean embedding of group)
    image_count     INTEGER
);
"""

_CREATE_IDX = """
CREATE INDEX IF NOT EXISTS idx_decisions_group ON decisions(group_id);
CREATE INDEX IF NOT EXISTS idx_decisions_created ON decisions(created_at);
"""


# ─── Memory record ────────────────────────────────────────────────────────────
class DecisionRecord:
    def __init__(
        self,
        group_id:       str,
        tier:           str,
        kept_path:      str,
        deleted_paths:  list[str],
        reason_tags:    list[str],
        human_notes:    str,
        llm_suggestion: Optional[str],
        llm_confidence: float,
        human_accepted: bool,
        quality_scores: dict[str, float],
        group_embedding: Optional[list[float]] = None,
        image_count:    int = 0,
    ):
        self.id             = str(uuid.uuid4())
        self.created_at     = datetime.utcnow().isoformat()
        self.group_id       = group_id
        self.tier           = tier
        self.kept_path      = kept_path
        self.deleted_paths  = deleted_paths
        self.reason_tags    = reason_tags
        self.human_notes    = human_notes
        self.llm_suggestion = llm_suggestion
        self.llm_confidence = llm_confidence
        self.human_accepted = human_accepted
        self.quality_scores = quality_scores
        self.group_embedding = group_embedding
        self.image_count    = image_count


# ─── Store ────────────────────────────────────────────────────────────────────
class RAGMemoryStore:
    """
    Two-layer memory:
      1. SQLite — full decision history, queryable
      2. ChromaDB — vector search for similar past groups (if available)

    Falls back to SQLite-only if ChromaDB is not installed.
    """

    def __init__(self):
        self._init_sqlite()
        self._init_chroma()

    # ── Public API ────────────────────────────────────────────────────────────
    def save_decision(self, record: DecisionRecord) -> str:
        """Persist a decision. Returns the record id."""
        self._sqlite_insert(record)
        self._chroma_insert(record)
        logger.info(f"Decision saved: {record.id} (kept {Path(record.kept_path).name})")
        return record.id

    def retrieve_similar(
        self,
        group_embedding: Optional[list[float]],
        k: int = RAG_TOP_K_MEMORIES,
    ) -> list[dict]:
        """
        Return up to k past decisions similar to the current group.
        Uses ChromaDB vector search when available, else returns recents.
        """
        if self._chroma_col is not None and group_embedding:
            return self._chroma_query(group_embedding, k)
        return self._sqlite_recent(k)

    def get_all_decisions(self, limit: int = 1000) -> list[dict]:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM decisions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        con.close()
        return [self._row_to_dict(r) for r in rows]

    def get_stats(self) -> dict:
        con = sqlite3.connect(DB_PATH)
        stats = {}
        stats["total_decisions"] = con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        stats["accepted"]        = con.execute("SELECT COUNT(*) FROM decisions WHERE human_accepted=1").fetchone()[0]
        stats["rejected"]        = con.execute("SELECT COUNT(*) FROM decisions WHERE human_accepted=0").fetchone()[0]
        stats["total_deleted"]   = sum(
            len(json.loads(r[0]))
            for r in con.execute("SELECT deleted_paths FROM decisions")
        )
        con.close()
        return stats

    # ── SQLite ────────────────────────────────────────────────────────────────
    def _init_sqlite(self):
        con = sqlite3.connect(DB_PATH)
        con.executescript(_CREATE_TABLE + _CREATE_IDX)
        con.commit()
        con.close()
        logger.info(f"SQLite memory store ready at {DB_PATH}")

    def _sqlite_insert(self, rec: DecisionRecord):
        con = sqlite3.connect(DB_PATH)
        con.execute(
            """INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rec.id, rec.created_at, rec.group_id, rec.tier,
                rec.kept_path,
                json.dumps(rec.deleted_paths),
                json.dumps(rec.reason_tags),
                rec.human_notes,
                rec.llm_suggestion,
                rec.llm_confidence,
                int(rec.human_accepted),
                json.dumps(rec.quality_scores),
                json.dumps(rec.group_embedding) if rec.group_embedding else None,
                rec.image_count,
            )
        )
        con.commit()
        con.close()

    def _sqlite_recent(self, k: int) -> list[dict]:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM decisions ORDER BY created_at DESC LIMIT ?", (k,)
        ).fetchall()
        con.close()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row) -> dict:
        d = dict(row)
        for key in ("deleted_paths", "reason_tags", "quality_scores", "group_embedding"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except Exception:
                    pass
        return d

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    def _init_chroma(self):
        self._chroma_col = None
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self._chroma_col = client.get_or_create_collection(
                name="photo_decisions",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB ready at {CHROMA_DIR} ({self._chroma_col.count()} records)")
        except Exception as e:
            logger.warning(f"ChromaDB unavailable ({e}), using SQLite-only retrieval")

    def _chroma_insert(self, rec: DecisionRecord):
        if self._chroma_col is None or not rec.group_embedding:
            return
        try:
            meta = {
                "group_id":       rec.group_id,
                "tier":           rec.tier,
                "kept_path":      rec.kept_path,
                "reason_tags":    ", ".join(rec.reason_tags),
                "human_accepted": int(rec.human_accepted),
                "llm_confidence": rec.llm_confidence,
                "image_count":    rec.image_count,
                "created_at":     rec.created_at,
            }
            doc = self._record_to_doc(rec)
            self._chroma_col.add(
                ids=[rec.id],
                embeddings=[rec.group_embedding],
                metadatas=[meta],
                documents=[doc],
            )
        except Exception as e:
            logger.warning(f"ChromaDB insert failed: {e}")

    def _chroma_query(self, embedding: list[float], k: int) -> list[dict]:
        try:
            results = self._chroma_col.query(
                query_embeddings=[embedding],
                n_results=min(k, max(1, self._chroma_col.count())),
                include=["metadatas", "documents", "distances"],
            )
            out = []
            for meta, doc, dist in zip(
                results["metadatas"][0],
                results["documents"][0],
                results["distances"][0],
            ):
                out.append({**meta, "document": doc, "similarity": round(1 - dist, 4)})
            return out
        except Exception as e:
            logger.warning(f"ChromaDB query failed: {e}")
            return self._sqlite_recent(k)

    @staticmethod
    def _record_to_doc(rec: DecisionRecord) -> str:
        tags  = ", ".join(rec.reason_tags) if rec.reason_tags else "none"
        acc   = "accepted" if rec.human_accepted else "rejected"
        return (
            f"Group tier={rec.tier}, {rec.image_count} photos. "
            f"Kept: {Path(rec.kept_path).name}. "
            f"Reason tags: {tags}. "
            f"LLM suggestion was {'correct' if rec.human_accepted else 'wrong'}. "
            f"Human {acc} the suggestion. Notes: {rec.human_notes or 'none'}."
        )
