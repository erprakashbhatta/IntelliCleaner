"""
doc_grouper.py — Document Duplicate Grouper
Two-tier duplicate detection for PDF and Word documents:
  Tier 1 — SHA-256 exact file hash  (binary identical)
  Tier 2 — TF-IDF cosine text similarity  (same/similar content)
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from doc_scanner import DocumentRecord

logger = logging.getLogger(__name__)

DOC_TEXT_SIMILARITY_THRESHOLD = 0.85   # cosine similarity ≥ this → near-duplicate


@dataclass
class DocumentDuplicateGroup:
    group_id:   str
    tier:       str          # "exact" | "near"
    similarity: float        # 0.0–1.0
    docs:       list[DocumentRecord]
    decided:    bool = False
    kept_idx:   Optional[int] = None
    kept_path:  Optional[str] = None


class DocumentDuplicateGrouper:
    """
    Takes a flat list of DocumentRecords and returns DocumentDuplicateGroups.

    Usage:
        grouper = DocumentDuplicateGrouper(records)
        groups, singletons = grouper.group()
    """

    def __init__(self, records: list[DocumentRecord]):
        self.records = [r for r in records if not r.error]
        self._used: set[str] = set()

    def group(self) -> tuple[list[DocumentDuplicateGroup], list[DocumentRecord]]:
        groups: list[DocumentDuplicateGroup] = []
        idx = 0

        # ── Tier 1: Exact file hash ───────────────────────────────────────────
        hash_map: dict[str, list[DocumentRecord]] = defaultdict(list)
        for r in self.records:
            if r.file_hash:
                hash_map[r.file_hash].append(r)

        for recs in hash_map.values():
            if len(recs) > 1:
                g = DocumentDuplicateGroup(
                    group_id=f"doc_exact_{idx:05d}",
                    tier="exact",
                    similarity=1.0,
                    docs=recs,
                )
                groups.append(g)
                for r in recs:
                    self._used.add(r.path)
                idx += 1

        # ── Tier 2: Text content cosine similarity ────────────────────────────
        remaining = [
            r for r in self.records
            if r.path not in self._used and r.text_content.strip()
        ]
        if len(remaining) >= 2:
            for recs, sim in self._cluster_by_text(remaining):
                g = DocumentDuplicateGroup(
                    group_id=f"doc_near_{idx:05d}",
                    tier="near",
                    similarity=round(sim, 3),
                    docs=recs,
                )
                groups.append(g)
                for r in recs:
                    self._used.add(r.path)
                idx += 1

        singletons = [r for r in self.records if r.path not in self._used]
        logger.info(f"Document grouper: {len(groups)} duplicate groups, {len(singletons)} unique")
        return groups, singletons

    # ── Text clustering ───────────────────────────────────────────────────────
    def _cluster_by_text(
        self, records: list[DocumentRecord]
    ) -> list[tuple[list[DocumentRecord], float]]:
        texts = [r.text_content for r in records]
        try:
            vectorizer   = TfidfVectorizer(max_features=5000, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(texts)
        except Exception as e:
            logger.warning(f"TF-IDF vectorization failed: {e}")
            return []

        sim_matrix = cosine_similarity(tfidf_matrix)
        n          = len(records)
        visited    = [False] * n
        results: list[tuple[list[DocumentRecord], float]] = []

        for i in range(n):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True
            for j in range(i + 1, n):
                if not visited[j] and sim_matrix[i, j] >= DOC_TEXT_SIMILARITY_THRESHOLD:
                    cluster.append(j)
                    visited[j] = True

            if len(cluster) >= 2:
                recs = [records[k] for k in cluster]
                pairs = [(a, b) for x, a in enumerate(cluster) for b in cluster[x + 1:]]
                avg_sim = float(np.mean([sim_matrix[a, b] for a, b in pairs])) if pairs else DOC_TEXT_SIMILARITY_THRESHOLD
                results.append((recs, avg_sim))

        return results
