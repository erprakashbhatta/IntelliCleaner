"""
pipeline.py — Main Orchestration Pipeline
Ties scanner → embedder → quality scorer → grouper → LLM analyzer together.
Results are emitted as events for the web UI to consume via SocketIO.
"""
import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

from rich.console import Console

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import LLM_CONFIDENCE_AUTO_THRESHOLD, THUMBNAILS_DIR
from scanner   import ImageScanner, ImageRecord
from embedder  import CLIPEmbedder
from quality   import QualityScorer
from grouper   import DuplicateGrouper, DuplicateGroup
from llm_analyzer import LLMAnalyzer, AnalysisResult
from rag_store    import RAGMemoryStore, DecisionRecord

console = Console()
logger  = logging.getLogger(__name__)


class PipelineResult:
    def __init__(self):
        self.groups:     list[DuplicateGroup] = []
        self.singletons: list[ImageRecord]    = []
        self.analyses:   dict[str, AnalysisResult] = {}   # group_id → result
        self.scan_time:  float = 0.0
        self.total_images: int = 0
        self.total_groups: int = 0


class Pipeline:
    """
    Full photo deduplication pipeline.

    Usage:
        pipeline = Pipeline("/path/to/photos")
        result   = pipeline.run(progress_cb=my_callback)
        # then handle result.groups one by one
    """

    def __init__(self, folder: str, excluded_dirs: set | None = None):
        self.folder        = folder
        self.excluded_dirs = excluded_dirs or set()
        self.memory        = RAGMemoryStore()
        self.embedder      = CLIPEmbedder()
        self.scorer        = QualityScorer()
        self.analyzer      = LLMAnalyzer(self.memory)

    # ── Main run ──────────────────────────────────────────────────────────────
    def run(
        self,
        progress_cb: Optional[Callable[..., None]] = None,
    ) -> PipelineResult:
        def emit(msg: str, pct: int = 0, scanned: int | None = None, total: int | None = None):
            console.print(f"  [cyan]{msg}[/]")
            if progress_cb:
                progress_cb(msg, pct, scanned, total)

        result = PipelineResult()
        t0     = datetime.utcnow()

        emit("Scanning folder for images...", 5, 0, 0)
        scanner = ImageScanner(self.folder, excluded_dirs=self.excluded_dirs)
        records = scanner.scan(progress_cb=emit)
        result.total_images = len(records)

        emit("Generating visual embeddings (CLIP)...", 20)
        records = self.embedder.embed(records)

        emit("Scoring photo quality...", 45)
        records = self.scorer.score(records)

        emit("Grouping duplicate photos...", 60)
        grouper  = DuplicateGrouper(records)
        groups, singletons = grouper.group()

        emit(f"Analysing {len(groups)} duplicate groups with LLM...", 70)
        analyses = {}
        for i, grp in enumerate(groups):
            pct = 70 + int(25 * i / max(len(groups), 1))
            emit(f"  Analysing group {i+1}/{len(groups)} ({grp.tier})...", pct)
            try:
                analysis = self.analyzer.analyze(grp)
                grp.best_guess_idx = analysis.best_idx
                analyses[grp.group_id] = analysis
            except Exception as e:
                logger.warning(f"Analysis failed for {grp.group_id}: {e}")

        result.groups     = groups
        result.singletons = singletons
        result.analyses   = analyses
        result.total_groups = len(groups)
        result.scan_time  = (datetime.utcnow() - t0).total_seconds()

        emit(f"Done - {len(groups)} groups found in {result.scan_time:.1f}s", 100)
        return result

    # ── Decision handling ─────────────────────────────────────────────────────
    def apply_decision(
        self,
        group: DuplicateGroup,
        kept_idx: int,
        reason_tags: list[str],
        human_notes: str,
        output_folder: str,
        dry_run: bool = False,
    ) -> dict:
        """
        Move the 'kept' image to output_folder.
        Move all others to output_folder/review/.
        Save the decision to RAG memory.
        Returns a summary dict.
        """
        out_dir    = Path(output_folder)
        review_dir = out_dir / "review"
        out_dir.mkdir(parents=True, exist_ok=True)
        review_dir.mkdir(parents=True, exist_ok=True)

        kept_rec  = group.images[kept_idx]
        other_recs = [r for i, r in enumerate(group.images) if i != kept_idx]

        moved_kept   = None
        moved_others = []

        if not dry_run:
            # Copy best photo to output
            dest = out_dir / kept_rec.filename
            dest = self._unique_path(dest)
            shutil.copy2(kept_rec.path, dest)
            moved_kept = str(dest)

            # Move rest to /review
            for r in other_recs:
                rdest = review_dir / r.filename
                rdest = self._unique_path(rdest)
                shutil.copy2(r.path, rdest)
                moved_others.append(str(rdest))

        # Was LLM suggestion accepted?
        llm_suggestion_path = None
        llm_confidence      = 0.0
        analysis = self.pipeline_result_analyses_ref.get(group.group_id) if hasattr(self, "pipeline_result_analyses_ref") else None
        if analysis:
            si = analysis.best_idx
            if 0 <= si < len(group.images):
                llm_suggestion_path = group.images[si].path
            llm_confidence = analysis.confidence

        human_accepted = (llm_suggestion_path == kept_rec.path)

        # Quality scores map
        quality_scores = {
            r.path: r.quality_score or 0.0 for r in group.images
        }

        # Group embedding (mean)
        group_emb = None
        embs = [r.embedding for r in group.images if r.embedding]
        if embs:
            import numpy as np
            group_emb = np.mean(np.array(embs, dtype=np.float32), axis=0).tolist()

        # Save to memory
        dec = DecisionRecord(
            group_id        = group.group_id,
            tier            = group.tier,
            kept_path       = kept_rec.path,
            deleted_paths   = [r.path for r in other_recs],
            reason_tags     = reason_tags,
            human_notes     = human_notes,
            llm_suggestion  = llm_suggestion_path,
            llm_confidence  = llm_confidence,
            human_accepted  = human_accepted,
            quality_scores  = quality_scores,
            group_embedding = group_emb,
            image_count     = len(group.images),
        )
        self.memory.save_decision(dec)

        group.human_choice_idx  = kept_idx
        group.human_reason_tags = reason_tags
        group.human_notes       = human_notes
        group.decided           = True

        return {
            "decision_id":   dec.id,
            "kept":          moved_kept or kept_rec.path,
            "review":        moved_others,
            "llm_accepted":  human_accepted,
            "dry_run":       dry_run,
        }

    @staticmethod
    def _unique_path(p: Path) -> Path:
        if not p.exists():
            return p
        stem, suffix = p.stem, p.suffix
        i = 1
        while True:
            candidate = p.with_name(f"{stem}_{i}{suffix}")
            if not candidate.exists():
                return candidate
            i += 1
