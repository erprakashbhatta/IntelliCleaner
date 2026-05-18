"""
core/grouper.py — Duplicate Grouper
Three-tier duplicate detection:
  Tier 1 — SHA-256 exact file hash
  Tier 2 — Perceptual hash (pHash Hamming distance)
  Tier 3 — CLIP semantic embedding cosine similarity
  Bonus  — Burst detection (same camera, taken within N seconds)
"""
import logging
from datetime import timedelta
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PHASH_EXACT_THRESHOLD, PHASH_NEAR_THRESHOLD,
    CLIP_SEMANTIC_THRESHOLD, BURST_TIME_WINDOW_SEC,
    MAX_SEMANTIC_CLUSTER_SIZE,
)
from scanner import ImageRecord

logger = logging.getLogger(__name__)


# ─── Data model ───────────────────────────────────────────────────────────────
@dataclass
class DuplicateGroup:
    group_id:      str
    tier:          str          # "exact" | "near" | "semantic" | "burst"
    similarity:    float        # 0.0–1.0
    images:        list[ImageRecord]
    best_guess_idx: Optional[int] = None     # LLM/scorer suggestion
    human_choice_idx: Optional[int] = None  # final human decision
    human_reason_tags: list[str] = field(default_factory=list)
    human_notes:   str = ""
    decided:       bool = False


# ─── Grouper ──────────────────────────────────────────────────────────────────
class DuplicateGrouper:
    """
    Takes a flat list of ImageRecords and returns DuplicateGroups.
    Records not in any group are returned separately as unique images.
    """

    def __init__(self, records: list[ImageRecord]):
        self.records = [r for r in records if not r.error]
        self._used: set[str] = set()

    def group(self) -> tuple[list[DuplicateGroup], list[ImageRecord]]:
        """
        Returns (groups, singletons).
        groups    — list of DuplicateGroup, each with 2+ images
        singletons — images not belonging to any group
        """
        groups: list[DuplicateGroup] = []
        idx    = 0

        # ── Tier 1: Exact hash ────────────────────────────────────────────────
        hash_map: dict[str, list[ImageRecord]] = defaultdict(list)
        for r in self.records:
            hash_map[r.file_hash].append(r)
        for fhash, recs in hash_map.items():
            if len(recs) > 1:
                g = DuplicateGroup(
                    group_id=f"exact_{idx:05d}",
                    tier="exact",
                    similarity=1.0,
                    images=recs,
                )
                groups.append(g)
                for r in recs:
                    self._used.add(r.path)
                idx += 1

        # ── Tier 2: Perceptual hash ───────────────────────────────────────────
        remaining = [r for r in self.records if r.path not in self._used and r.phash]
        phash_groups = self._cluster_phash(remaining)
        for recs in phash_groups:
            if len(recs) > 1:
                avg_dist = self._avg_phash_distance(recs)
                sim = max(0.0, 1.0 - avg_dist / 64.0)
                g = DuplicateGroup(
                    group_id=f"near_{idx:05d}",
                    tier="near",
                    similarity=round(sim, 3),
                    images=recs,
                )
                groups.append(g)
                for r in recs:
                    self._used.add(r.path)
                idx += 1

        # ── Tier 3: CLIP semantic ─────────────────────────────────────────────
        remaining = [r for r in self.records if r.path not in self._used and r.embedding]
        sem_groups = self._cluster_semantic(remaining)
        for recs, sim in sem_groups:
            if len(recs) > 1:
                g = DuplicateGroup(
                    group_id=f"semantic_{idx:05d}",
                    tier="semantic",
                    similarity=round(sim, 3),
                    images=recs,
                )
                groups.append(g)
                for r in recs:
                    self._used.add(r.path)
                idx += 1

        # ── Burst detection ───────────────────────────────────────────────────
        remaining = [r for r in self.records if r.path not in self._used]
        burst_groups = self._detect_bursts(remaining)
        for recs in burst_groups:
            if len(recs) > 1:
                g = DuplicateGroup(
                    group_id=f"burst_{idx:05d}",
                    tier="burst",
                    similarity=0.75,
                    images=recs,
                )
                groups.append(g)
                for r in recs:
                    self._used.add(r.path)
                idx += 1

        singletons = [r for r in self.records if r.path not in self._used]
        logger.info(f"Found {len(groups)} duplicate groups, {len(singletons)} unique images")
        return groups, singletons

    # ── Tier 2 helpers — LSH pHash clustering ────────────────────────────────
    def _cluster_phash(self, records: list[ImageRecord]) -> list[list[ImageRecord]]:
        """
        Near-duplicate clustering via LSH on 64-bit pHashes.

        Uses 16 bands of 4 bits → ~99.98% recall for Hamming distance ≤ 13.
        O(n) bucket build + O(candidates) verification — much faster than
        sklearn NearestNeighbors for large n.
        """
        if len(records) < 2:
            return []

        hashes: list[int] = []
        valid:  list[ImageRecord] = []
        for r in records:
            if not r.phash:
                continue
            try:
                hashes.append(int(r.phash, 16))
                valid.append(r)
            except Exception:
                continue

        if len(valid) < 2:
            return []

        n = len(valid)
        # 16 bands × 4 bits = 64 bits; gives very high recall for dist ≤ 13
        NUM_BANDS  = 16
        BAND_BITS  = 4
        BAND_MASKS = [(0xF << (i * BAND_BITS)) for i in range(NUM_BANDS)]

        # Build buckets
        buckets: dict[tuple, list[int]] = defaultdict(list)
        for idx, h in enumerate(hashes):
            for band_idx, mask in enumerate(BAND_MASKS):
                bval = (h & mask) >> (band_idx * BAND_BITS)
                buckets[(band_idx, bval)].append(idx)

        # Collect candidate pairs from bucket collisions
        candidate_pairs: set[tuple[int, int]] = set()
        for members in buckets.values():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    if a > b:
                        a, b = b, a
                    candidate_pairs.add((a, b))

        # Verify with actual Hamming distance (XOR + popcount)
        adjacency: dict[int, set[int]] = defaultdict(set)
        for a, b in candidate_pairs:
            dist = bin(hashes[a] ^ hashes[b]).count("1")
            if dist <= PHASH_NEAR_THRESHOLD:
                adjacency[a].add(b)
                adjacency[b].add(a)

        # Connected components (BFS)
        visited = [False] * n
        clusters: list[list[ImageRecord]] = []
        for start in range(n):
            if visited[start] or not adjacency.get(start):
                continue
            cluster_idx: list[int] = []
            queue = [start]
            while queue:
                node = queue.pop()
                if visited[node]:
                    continue
                visited[node] = True
                cluster_idx.append(node)
                for nb in adjacency[node]:
                    if not visited[nb]:
                        queue.append(nb)
            if len(cluster_idx) >= 2:
                clusters.append([valid[i] for i in cluster_idx])

        return clusters

    @staticmethod
    def _avg_phash_distance(records: list[ImageRecord]) -> float:
        if len(records) < 2:
            return 0.0
        try:
            hashes = [int(r.phash, 16) for r in records if r.phash]
            dists  = [bin(hashes[i] ^ hashes[j]).count("1")
                      for i in range(len(hashes))
                      for j in range(i + 1, len(hashes))]
            return float(np.mean(dists)) if dists else 0.0
        except Exception:
            return 0.0

    # ── Tier 3 helpers ────────────────────────────────────────────────────────
    def _cluster_semantic(
        self, records: list[ImageRecord]
    ) -> list[tuple[list[ImageRecord], float]]:
        if len(records) > MAX_SEMANTIC_CLUSTER_SIZE:
            logger.warning(
                f"Skipping CLIP semantic clustering: {len(records)} records exceeds "
                f"MAX_SEMANTIC_CLUSTER_SIZE={MAX_SEMANTIC_CLUSTER_SIZE}. "
                f"Lower the threshold or increase the limit in config.py."
            )
            return []
        from sklearn.neighbors import NearestNeighbors
        if len(records) < 2:
            return []

        embs = np.array([r.embedding for r in records], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs = embs / norms

        radius = 1.0 - CLIP_SEMANTIC_THRESHOLD
        neigh = NearestNeighbors(radius=radius, metric="cosine", n_jobs=-1)
        neigh.fit(embs)
        graph = neigh.radius_neighbors_graph(embs)
        index_groups = self._indices_from_graph(graph)

        results = []
        for indices in index_groups:
            if len(indices) < 2:
                continue
            recs = [records[i] for i in indices]
            sub_embs = embs[indices]
            if sub_embs.shape[0] < 2:
                avg_sim = CLIP_SEMANTIC_THRESHOLD
            else:
                sims = sub_embs @ sub_embs.T
                i_upper, j_upper = np.triu_indices(sub_embs.shape[0], k=1)
                avg_sim = float(np.mean(sims[i_upper, j_upper])) if len(i_upper) else CLIP_SEMANTIC_THRESHOLD
            results.append((recs, avg_sim))

        return results

    # ── Burst detection ───────────────────────────────────────────────────────
    @staticmethod
    def _detect_bursts(records: list[ImageRecord]) -> list[list[ImageRecord]]:
        """Group photos taken within BURST_TIME_WINDOW_SEC of each other."""
        timed = [r for r in records if r.taken_at is not None]
        timed.sort(key=lambda r: r.taken_at)

        window = timedelta(seconds=BURST_TIME_WINDOW_SEC)
        groups: list[list[ImageRecord]] = []
        current: list[ImageRecord] = []

        for r in timed:
            if not current:
                current.append(r)
            elif r.taken_at - current[-1].taken_at <= window:
                current.append(r)
            else:
                if len(current) > 1:
                    groups.append(current)
                current = [r]
        if len(current) > 1:
            groups.append(current)

        return groups
