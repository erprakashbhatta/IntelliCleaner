"""
ui/server.py — Flask + SocketIO Web Server
Serves the review UI and exposes REST + WebSocket endpoints.
"""
import os
import json
import logging
import threading
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit as sio_emit
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import UI_HOST, UI_PORT, THUMBNAILS_DIR, THUMBNAIL_SIZE, REASON_TAGS, SCANS_DIR
from pipeline import Pipeline, PipelineResult
from rag_store import RAGMemoryStore
from grouper import DuplicateGroup
from doc_scanner import DocumentScanner, DocumentRecord
from doc_grouper import DocumentDuplicateGrouper, DocumentDuplicateGroup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── App setup ────────────────────────────────────────────────────────────────
app    = Flask(__name__, static_folder=None)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ─── Global state (single-user local tool) ────────────────────────────────────
_state = {
    "pipeline":  None,
    "result":    None,
    "folder":    None,
    "output":    None,
    "scan_id":   None,
    "running":   False,
    "current_group_idx": 0,
}
_memory = RAGMemoryStore()

# ─── Document scan state (separate from photo state) ──────────────────────────
_doc_state: dict = {
    "groups":    [],        # list[DocumentDuplicateGroup]
    "singletons": [],       # list[DocumentRecord]
    "folder":    None,
    "running":   False,
    "total_docs": 0,
    "total_groups": 0,
}


# ─── Helper ───────────────────────────────────────────────────────────────────
def _json_safe(value):
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()

    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _group_to_dict(grp, analysis=None, idx: int = 0) -> dict:
    images = []
    for i, rec in enumerate(grp.images):
        thumb = _ensure_thumbnail(rec)
        images.append({
            "index":       i,
            "path":        rec.path,
            "filename":    rec.filename,
            "size_bytes":  rec.size_bytes,
            "width":       rec.width,
            "height":      rec.height,
            "taken_at":    rec.taken_at.isoformat() if rec.taken_at else None,
            "camera":      rec.camera,
            "quality_score":   float(rec.quality_score or 0),
            "quality_details": _json_safe(rec.quality_details or {}),
            "thumbnail":   f"/thumb/{Path(thumb).name}" if thumb else None,
            "llm_reason":  analysis.reasons.get(str(i), "") if analysis else "",
            "disqualifiers": analysis.disqualifiers.get(str(i), []) if analysis else [],
            "is_best_guess": (analysis.best_idx == i) if analysis else False,
        })

    return {
        "group_id":    grp.group_id,
        "index":       idx,
        "tier":        grp.tier,
        "similarity":  float(grp.similarity),
        "image_count": len(grp.images),
        "images":      images,
        "decided":     grp.decided,
        "llm_recommendation": analysis.recommendation if analysis else "",
        "llm_confidence":     float(analysis.confidence) if analysis else 0.0,
        "llm_backend":        analysis.backend if analysis else "none",
        "rag_context_used":   analysis.rag_context_used if analysis else 0,
        "best_guess_idx":     grp.best_guess_idx,
    }


def _ensure_thumbnail(rec) -> str | None:
    if rec.thumbnail_path:
        thumb_path = Path(rec.thumbnail_path)
        if thumb_path.exists():
            return str(thumb_path)

    try:
        img = Image.open(rec.path).convert("RGB")
        img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
        thumb_name = f"{Path(rec.path).stem}_{Path(rec.path).stat().st_size}.jpg"
        thumb_path = THUMBNAILS_DIR / thumb_name
        img.save(thumb_path, "JPEG", quality=85)
        rec.thumbnail_path = str(thumb_path)
        return str(thumb_path)
    except Exception:
        return None


def _save_scan_results(folder: str, result: PipelineResult):
    """Save scan results to disk for later reuse."""
    import json
    from datetime import datetime

    scan_id = f"{Path(folder).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    scan_dir = SCANS_DIR / scan_id
    scan_dir.mkdir(exist_ok=True)

    # Save metadata
    metadata = {
        "scan_id": scan_id,
        "folder": folder,
        "timestamp": datetime.now().isoformat(),
        "total_images": result.total_images,
        "total_groups": result.total_groups,
        "scan_time": result.scan_time,
    }

    with open(scan_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save records (without large embeddings to save space)
    records_data = []
    for rec in result.singletons + [img for group in result.groups for img in group.images]:
        record_dict = {
            "path": rec.path,
            "filename": rec.filename,
            "file_hash": rec.file_hash,
            "phash": rec.phash,
            "size_bytes": rec.size_bytes,
            "width": rec.width,
            "height": rec.height,
            "mode": rec.mode,
            "created_at": rec.created_at.isoformat() if rec.created_at else None,
            "taken_at": rec.taken_at.isoformat() if rec.taken_at else None,
            "camera": rec.camera,
            "gps_lat": rec.gps_lat,
            "gps_lon": rec.gps_lon,
            "thumbnail_path": rec.thumbnail_path,
            "quality_score": rec.quality_score,
            "quality_details": rec.quality_details,
            "error": rec.error,
            # Save embedding as list if it exists (handle both numpy arrays and plain lists)
            "embedding": rec.embedding.tolist() if rec.embedding is not None and hasattr(rec.embedding, 'tolist') else rec.embedding,
        }
        records_data.append(record_dict)

    with open(scan_dir / "records.json", "w") as f:
        json.dump(_json_safe(records_data), f, indent=2)

    # Save groups
    groups_data = []
    for group in result.groups:
        group_dict = {
            "group_id": group.group_id,
            "tier": group.tier,
            "similarity": group.similarity,
            "images": [rec.path for rec in group.images],
            "best_guess_idx": group.best_guess_idx,
            "decided": group.decided,
        }
        groups_data.append(group_dict)

    with open(scan_dir / "groups.json", "w") as f:
        json.dump(_json_safe(groups_data), f, indent=2)

    # Save analyses
    analyses_data = {}
    for group_id, analysis in result.analyses.items():
        analyses_data[group_id] = {
            "best_idx": analysis.best_idx,
            "confidence": analysis.confidence,
            "recommendation": analysis.recommendation,
            "reasons": analysis.reasons,
            "disqualifiers": analysis.disqualifiers,
            "backend": analysis.backend,
            "rag_context_used": analysis.rag_context_used,
        }

    with open(scan_dir / "analyses.json", "w") as f:
        json.dump(_json_safe(analyses_data), f, indent=2)

    return scan_id


def _load_scan_results(scan_id: str) -> tuple[str, PipelineResult] | None:
    """
    Load scan results from disk.
    Returns None (instead of raising) if any required file is missing or corrupt.
    """
    from datetime import datetime

    scan_dir = SCANS_DIR / scan_id
    if not scan_dir.exists():
        return None

    required = ["metadata.json", "records.json", "groups.json"]
    for fname in required:
        if not (scan_dir / fname).exists():
            logger.warning(f"Scan {scan_id} missing {fname} — skipping")
            return None

    try:
        with open(scan_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
    except Exception as e:
        logger.warning(f"Scan {scan_id}: corrupt metadata.json — {e}")
        return None

    try:
        with open(scan_dir / "records.json", "r") as f:
            records_data = json.load(f)
    except Exception as e:
        logger.warning(f"Scan {scan_id}: corrupt records.json — {e}")
        return None

    try:
        with open(scan_dir / "groups.json", "r") as f:
            groups_data = json.load(f)
    except Exception as e:
        logger.warning(f"Scan {scan_id}: corrupt groups.json — {e}")
        return None

    # Reconstruct ImageRecord objects
    records_dict = {}
    for rec_data in records_data:
        try:
            rec = ImageRecord(
                path=rec_data["path"],
                filename=rec_data["filename"],
                file_hash=rec_data.get("file_hash", ""),
                phash=rec_data.get("phash", ""),
                size_bytes=rec_data.get("size_bytes", 0),
                width=rec_data.get("width", 0),
                height=rec_data.get("height", 0),
                mode=rec_data.get("mode", ""),
                created_at=datetime.fromisoformat(rec_data["created_at"]) if rec_data.get("created_at") else None,
                taken_at=datetime.fromisoformat(rec_data["taken_at"])    if rec_data.get("taken_at")    else None,
                camera=rec_data.get("camera"),
                gps_lat=rec_data.get("gps_lat"),
                gps_lon=rec_data.get("gps_lon"),
                thumbnail_path=rec_data.get("thumbnail_path"),
                quality_score=rec_data.get("quality_score"),
                error=rec_data.get("error"),
            )
            rec.quality_details = rec_data.get("quality_details") or {}
            emb = rec_data.get("embedding")
            if emb:
                import numpy as np
                rec.embedding = np.array(emb)
            records_dict[rec.path] = rec
        except Exception as e:
            logger.debug(f"Scan {scan_id}: skipping corrupt record entry — {e}")
            continue

    # Reconstruct groups
    groups = []
    for gd in groups_data:
        try:
            images = [records_dict[p] for p in gd.get("images", []) if p in records_dict]
            if len(images) >= 2:
                groups.append(DuplicateGroup(
                    group_id=gd["group_id"],
                    tier=gd.get("tier", "exact"),
                    similarity=gd.get("similarity", 1.0),
                    images=images,
                    best_guess_idx=gd.get("best_guess_idx"),
                    decided=gd.get("decided", False),
                ))
        except Exception as e:
            logger.debug(f"Scan {scan_id}: skipping corrupt group entry — {e}")
            continue

    # Load analyses (optional — failures are non-fatal)
    analyses = {}
    analyses_file = scan_dir / "analyses.json"
    if analyses_file.exists():
        try:
            with open(analyses_file, "r") as f:
                analyses_data = json.load(f)
            for group_id, ad in analyses_data.items():
                try:
                    from llm_analyzer import AnalysisResult
                    analyses[group_id] = AnalysisResult(
                        best_idx=ad["best_idx"],
                        confidence=ad.get("confidence", 0.0),
                        recommendation=ad.get("recommendation", ""),
                        reasons=ad.get("reasons", {}),
                        disqualifiers=ad.get("disqualifiers", {}),
                        backend=ad.get("backend", "none"),
                        rag_context_used=ad.get("rag_context_used", 0),
                    )
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Scan {scan_id}: could not load analyses.json — {e}")

    grouped_paths = {img.path for g in groups for img in g.images}
    singletons    = [r for r in records_dict.values() if r.path not in grouped_paths]

    result               = PipelineResult()
    result.groups        = groups
    result.singletons    = singletons
    result.analyses      = analyses
    result.scan_time     = metadata.get("scan_time", 0.0)
    result.total_images  = metadata.get("total_images", len(records_dict))
    result.total_groups  = metadata.get("total_groups", len(groups))

    return metadata["folder"], result


def _list_saved_scans():
    """List all saved scan results."""
    import json
    from datetime import datetime

    scans = []
    if SCANS_DIR.exists():
        for scan_dir in SCANS_DIR.iterdir():
            if scan_dir.is_dir():
                metadata_file = scan_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        scans.append(metadata)
                    except:
                        continue

    # Sort by timestamp (newest first)
    scans.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return scans


# ─── REST endpoints ───────────────────────────────────────────────────────────
@app.route("/api/scan", methods=["POST"])
def api_scan():
    data   = request.json or {}
    folder = data.get("folder", "").strip()
    output = data.get("output", "").strip()
    # Extra folder names (not full paths) to exclude from this scan
    extra_excluded = set(data.get("excluded_dirs", []))

    if not folder or not Path(folder).is_dir():
        return jsonify({"error": "Invalid folder path"}), 400

    if not output:
        output = str(Path(folder).parent / (Path(folder).name + "_deduplicated"))

    if _state["running"]:
        return jsonify({"error": "Scan already in progress"}), 409

    _state["folder"]  = folder
    _state["output"]  = output
    _state["running"] = True
    _state["result"]  = None
    _state["current_group_idx"] = 0

    def run_pipeline():
        def progress_cb(msg: str, pct: int, scanned: int | None = None, total: int | None = None):
            payload = {"message": msg, "percent": pct}
            if scanned is not None and total is not None:
                payload["scanned"] = scanned
                payload["total"] = total
            socketio.emit("progress", payload)

        try:
            pipe   = Pipeline(folder, excluded_dirs=extra_excluded)
            result = pipe.run(progress_cb=progress_cb)

            # Store reference so apply_decision works
            pipe.pipeline_result_analyses_ref = result.analyses

            _state["pipeline"] = pipe
            _state["result"]   = result
            _state["running"]  = False

            # Auto-save the scan results
            try:
                scan_id = _save_scan_results(folder, result)
                _state["scan_id"] = scan_id
                logger.info(f"Scan auto-saved as {scan_id}")
            except Exception as e:
                logger.warning(f"Failed to auto-save scan: {e}")

            # Send summary to UI (include scan_id so JS can track it)
            socketio.emit("scan_complete", {
                "total_images":  result.total_images,
                "total_groups":  result.total_groups,
                "total_singles": len(result.singletons),
                "scan_time":     result.scan_time,
                "scan_id":       _state.get("scan_id"),
            })
        except Exception as e:
            logger.exception("Pipeline error")
            _state["running"] = False
            socketio.emit("scan_error", {"error": str(e)})

    threading.Thread(target=run_pipeline, daemon=True).start()
    return jsonify({"status": "started", "folder": folder, "output": output})


@app.route("/api/groups", methods=["GET"])
def api_groups():
    result: PipelineResult = _state.get("result")
    if not result:
        return jsonify({"error": "No scan results yet"}), 404

    tier   = request.args.get("tier")
    decided = request.args.get("decided")

    groups = result.groups
    if tier:
        groups = [g for g in groups if g.tier == tier]
    if decided == "false":
        groups = [g for g in groups if not g.decided]
    elif decided == "true":
        groups = [g for g in groups if g.decided]

    out = []
    for i, g in enumerate(groups):
        analysis = result.analyses.get(g.group_id)
        out.append(_group_to_dict(g, analysis, i))

    return jsonify({
        "groups": out,
        "total":  len(out),
        "undecided": sum(1 for g in result.groups if not g.decided),
    })


@app.route("/api/groups/<group_id>", methods=["GET"])
def api_group_detail(group_id: str):
    result: PipelineResult = _state.get("result")
    if not result:
        return jsonify({"error": "No scan results"}), 404

    for i, g in enumerate(result.groups):
        if g.group_id == group_id:
            analysis = result.analyses.get(g.group_id)
            return jsonify(_group_to_dict(g, analysis, i))
    return jsonify({"error": "Group not found"}), 404


@app.route("/api/decide", methods=["POST"])
def api_decide():
    data = request.json or {}
    group_id   = data.get("group_id", "")
    kept_idx   = data.get("kept_idx")
    reason_tags = data.get("reason_tags", [])
    human_notes = data.get("human_notes", "")
    dry_run     = data.get("dry_run", False)

    if kept_idx is None:
        return jsonify({"error": "kept_idx required"}), 400

    result:   PipelineResult = _state.get("result")
    pipeline: Pipeline       = _state.get("pipeline")
    output    = _state.get("output", "")

    if not result or not pipeline:
        return jsonify({"error": "No scan results"}), 404

    grp = next((g for g in result.groups if g.group_id == group_id), None)
    if not grp:
        return jsonify({"error": "Group not found"}), 404

    if not (0 <= kept_idx < len(grp.images)):
        return jsonify({"error": "Invalid kept_idx"}), 400

    try:
        summary = pipeline.apply_decision(
            group=grp,
            kept_idx=kept_idx,
            reason_tags=reason_tags,
            human_notes=human_notes,
            output_folder=output,
            dry_run=dry_run,
        )
        undecided = sum(1 for g in result.groups if not g.decided)
        summary["undecided_remaining"] = undecided
        return jsonify(summary)
    except Exception as e:
        logger.exception("Decision error")
        return jsonify({"error": str(e)}), 500


# ─── Scan Management endpoints ──────────────────────────────────────────────────
@app.route("/api/scans/save", methods=["POST"])
def api_save_scan():
    result: PipelineResult = _state.get("result")
    folder = _state.get("folder")

    if not result or not folder:
        return jsonify({"error": "No scan results to save"}), 404

    try:
        scan_id = _save_scan_results(folder, result)
        return jsonify({"scan_id": scan_id, "message": "Scan saved successfully"})
    except Exception as e:
        logger.exception("Save scan error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/scans", methods=["GET"])
def api_list_scans():
    try:
        scans = _list_saved_scans()
        return jsonify({"scans": scans})
    except Exception as e:
        logger.exception("List scans error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/scans/<scan_id>", methods=["POST"])
def api_load_scan(scan_id: str):
    if _state["running"]:
        return jsonify({"error": "Scan already in progress"}), 409

    try:
        loaded = _load_scan_results(scan_id)
        if not loaded:
            return jsonify({"error": "Scan not found"}), 404

        folder, result = loaded
        _state["folder"] = folder
        _state["result"] = result
        _state["scan_id"] = scan_id
        _state["current_group_idx"] = 0

        # Create a dummy pipeline object for decision handling
        from pipeline import Pipeline
        pipe = Pipeline(folder)
        pipe.pipeline_result_analyses_ref = result.analyses
        _state["pipeline"] = pipe

        return jsonify({
            "folder": folder,
            "total_images": result.total_images,
            "total_groups": result.total_groups,
            "scan_time": result.scan_time,
            "scan_id": scan_id,
            "message": "Scan loaded successfully"
        })
    except Exception as e:
        logger.exception("Load scan error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/scans/<scan_id>", methods=["DELETE"])
def api_delete_scan(scan_id: str):
    try:
        scan_dir = SCANS_DIR / scan_id
        if scan_dir.exists():
            import shutil
            shutil.rmtree(scan_dir)
            return jsonify({"message": "Scan deleted successfully"})
        else:
            return jsonify({"error": "Scan not found"}), 404
    except Exception as e:
        logger.exception("Delete scan error")
        return jsonify({"error": str(e)}), 500


# ─── Apply decisions endpoints ─────────────────────────────────────────────────
@app.route("/api/apply_decisions", methods=["POST"])
def api_apply_decisions():
    """Apply all made decisions: copy kept files, delete duplicates, generate report."""
    data = request.json or {}
    scan_id = data.get("scan_id")
    target_folder = data.get("target_folder")
    dry_run = data.get("dry_run", False)
    
    if not scan_id:
        return jsonify({"error": "scan_id required"}), 400
    
    try:
        from apply_decisions import DecisionApplier
        
        applier = DecisionApplier(scan_id, target_folder)
        applier.apply_decisions(dry_run=dry_run)
        report = applier.generate_report()
        
        return jsonify({
            "success": True,
            "dry_run": dry_run,
            "report": report,
            "message": f"{'Dry-run: ' if dry_run else ''}Applied decisions - {report['stats']['transferred_count']} files transferred, {report['stats']['deleted_count']} deleted"
        })
    
    except Exception as e:
        logger.exception("Apply decisions error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/apply_decisions/report/<scan_id>", methods=["GET"])
def api_get_decision_report(scan_id: str):
    """Get the latest decision report for a scan."""
    try:
        scan_dir = SCANS_DIR / scan_id
        if not scan_dir.exists():
            return jsonify({"error": "Scan not found"}), 404
        
        # Find the latest report file
        report_files = list(scan_dir.glob("decision_report_*.json"))
        if not report_files:
            return jsonify({"error": "No decision report found"}), 404
        
        latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_report) as f:
            report = json.load(f)
        
        return jsonify(report)
    
    except Exception as e:
        logger.exception("Get report error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/decide/auto", methods=["POST"])
def api_decide_auto():
    """Auto-apply all high-confidence LLM suggestions."""
    data      = request.json or {}
    threshold = float(data.get("confidence_threshold", 0.92))
    dry_run   = data.get("dry_run", False)

    result:   PipelineResult = _state.get("result")
    pipeline: Pipeline       = _state.get("pipeline")
    output    = _state.get("output", "")

    if not result or not pipeline:
        return jsonify({"error": "No scan results"}), 404

    applied = 0
    skipped = 0
    for grp in result.groups:
        if grp.decided:
            continue
        analysis = result.analyses.get(grp.group_id)
        if not analysis or analysis.confidence < threshold:
            skipped += 1
            continue
        try:
            pipeline.apply_decision(
                group=grp,
                kept_idx=analysis.best_idx,
                reason_tags=["auto_decision"],
                human_notes=f"Auto-applied (confidence={analysis.confidence:.2f})",
                output_folder=output,
                dry_run=dry_run,
            )
            applied += 1
        except Exception as e:
            logger.warning(f"Auto-decision failed for {grp.group_id}: {e}")
            skipped += 1

    return jsonify({"applied": applied, "skipped": skipped, "dry_run": dry_run})


@app.route("/api/transfer_files", methods=["POST"])
def api_transfer_files():
    """
    Copy every file from source_folder to target_folder preserving the
    full sub-directory structure.  Emits transfer_progress SocketIO events.
    Skips files already present in target (unless overwrite=true).
    """
    import shutil
    from datetime import datetime as _dt

    data          = request.json or {}
    source_folder = (data.get("source_folder") or _state.get("folder") or "").strip()
    target_folder = (data.get("target_folder") or "").strip()
    overwrite     = data.get("overwrite", False)

    if not source_folder or not Path(source_folder).is_dir():
        return jsonify({"error": "Invalid or missing source folder"}), 400
    if not target_folder:
        return jsonify({"error": "Target folder is required"}), 400

    src_root = Path(source_folder)
    tgt_root = Path(target_folder)

    def run_transfer():
        all_files = [p for p in src_root.rglob("*") if p.is_file()]
        total     = len(all_files)
        copied = skipped = errors = 0

        socketio.emit("transfer_progress", {
            "message": f"Starting transfer of {total} files…",
            "percent": 0, "copied": 0, "total": total,
        })

        for i, src in enumerate(all_files, 1):
            try:
                rel  = src.relative_to(src_root)
                dest = tgt_root / rel
                if dest.exists() and not overwrite:
                    skipped += 1
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(dest))
                    copied += 1
            except Exception as e:
                errors += 1
                logger.warning(f"Transfer error {src}: {e}")

            if i % 100 == 0 or i == total:
                socketio.emit("transfer_progress", {
                    "message": f"Transferred {copied} / {total} files…",
                    "percent": int(100 * i / total),
                    "copied":  copied, "total": total,
                })

        socketio.emit("transfer_complete", {
            "total":   total,
            "copied":  copied,
            "skipped": skipped,
            "errors":  errors,
            "target":  target_folder,
            "timestamp": _dt.now().isoformat(),
        })

    threading.Thread(target=run_transfer, daemon=True).start()
    return jsonify({"status": "started", "source": source_folder, "target": target_folder})


@app.route("/api/cache_stats", methods=["GET"])
def api_cache_stats():
    """Return scan cache statistics."""
    try:
        from scan_cache import get_cache
        stats = get_cache().stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cache_evict", methods=["POST"])
def api_cache_evict():
    """Remove cache entries for files that no longer exist on disk."""
    try:
        from scan_cache import get_cache
        removed = get_cache().evict_missing()
        return jsonify({"removed": removed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def api_stats():
    result = _state.get("result")
    mem_stats = _memory.get_stats()
    out = {**mem_stats}
    if result:
        out.update({
            "current_scan_images": result.total_images,
            "current_scan_groups": result.total_groups,
            "current_scan_decided": sum(1 for g in result.groups if g.decided),
        })
    return jsonify(out)


@app.route("/api/memory", methods=["GET"])
def api_memory():
    limit   = int(request.args.get("limit", 50))
    records = _memory.get_all_decisions(limit=limit)
    return jsonify({"decisions": records, "count": len(records)})


@app.route("/api/reason_tags", methods=["GET"])
def api_reason_tags():
    return jsonify({"tags": REASON_TAGS})


@app.route("/api/status", methods=["GET"])
def api_status():
    result = _state.get("result")
    return jsonify({
        "running":  _state["running"],
        "folder":   _state["folder"],
        "output":   _state["output"],
        "scan_id":  _state.get("scan_id"),
        "has_result": result is not None,
        "groups":   result.total_groups if result else 0,
        "undecided": sum(1 for g in result.groups if not g.decided) if result else 0,
    })


# ─── File serving ─────────────────────────────────────────────────────────────
@app.route("/thumb/<filename>")
def serve_thumb(filename: str):
    return send_from_directory(THUMBNAILS_DIR, filename)


@app.route("/image")
def serve_image():
    """Serve original image by path (local only, with path validation)."""
    path  = request.args.get("path", "")
    folder = _state.get("folder", "")
    if not path or not folder:
        return "Not found", 404
    p = Path(path)
    # Security: ensure path is under the scanned folder
    try:
        p.resolve().relative_to(Path(folder).resolve())
    except ValueError:
        return "Forbidden", 403
    if not p.exists():
        return "Not found", 404
    return send_file(str(p))


# ─── Photo apply-decisions (auto + manual, no scan_id required) ──────────────

@app.route("/api/apply_photo_decisions", methods=["POST"])
def api_apply_photo_decisions():
    """
    Move duplicate photos to a target folder and delete them from the source.

    Accepts:
      scan_id       – optional; defaults to the current in-memory scan or most-recent saved scan
      target_folder – optional; defaults to <source>_duplicates
      dry_run       – bool, default False

    For every duplicate group the best image is kept in place.
    Selection priority: highest quality_score → largest file size.
    Manual decisions (human_choice_idx set on the group) take precedence.
    """
    from datetime import datetime as _dt
    import shutil

    data          = request.json or {}
    scan_id       = (data.get("scan_id") or "").strip() or _state.get("scan_id")
    target_folder = (data.get("target_folder") or "").strip()
    dry_run       = data.get("dry_run", False)

    # ── Resolve which result to use ───────────────────────────────────────────
    result: PipelineResult | None = _state.get("result")

    if not result and not scan_id:
        # Try most-recent saved scan
        scans = _list_saved_scans()
        if scans:
            scan_id = scans[0].get("scan_id", "")

    if not result and scan_id:
        loaded = _load_scan_results(scan_id)
        if loaded:
            _, result = loaded

    if not result:
        return jsonify({"error": "No scan results found — run a scan first or load a saved scan"}), 404

    groups: list = result.groups
    if not groups:
        return jsonify({"error": "No duplicate groups in this scan"}), 404

    # ── Resolve target folder ─────────────────────────────────────────────────
    source_folder = _state.get("folder") or ""
    if not source_folder and scan_id:
        loaded = _load_scan_results(scan_id)
        if loaded:
            source_folder = loaded[0]

    if not target_folder:
        if source_folder:
            target_folder = str(Path(source_folder).parent / (Path(source_folder).name + "_duplicates"))
        else:
            return jsonify({"error": "Could not determine target folder — please provide one"}), 400

    target_path  = Path(target_folder)
    moved:  list[dict] = []
    errors: list[dict] = []
    auto_decided   = 0
    manual_decided = 0

    for grp in groups:
        images = grp.images

        # Determine keeper index
        if grp.human_choice_idx is not None:
            kept_idx = grp.human_choice_idx
            manual_decided += 1
        elif grp.best_guess_idx is not None:
            kept_idx = grp.best_guess_idx
            auto_decided += 1
        else:
            # Fallback: highest quality_score then largest size
            kept_idx = max(
                range(len(images)),
                key=lambda i: (images[i].quality_score or 0.0, images[i].size_bytes),
            )
            auto_decided += 1

        duplicates = [rec for i, rec in enumerate(images) if i != kept_idx]
        for rec in duplicates:
            src = Path(rec.path)
            if not src.exists():
                continue
            dest = target_path / src.name
            counter = 1
            while dest.exists() or any(m["to"] == str(dest) for m in moved):
                dest = target_path / f"{src.stem}_{counter}{src.suffix}"
                counter += 1
            try:
                if not dry_run:
                    target_path.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(dest))
                    src.unlink()
                moved.append({
                    "from":  str(src),
                    "to":    str(dest),
                    "kept":  images[kept_idx].filename,
                    "auto":  grp.human_choice_idx is None,
                })
            except Exception as e:
                errors.append({"file": str(rec.path), "error": str(e)})
                logger.warning(f"Failed to move photo {rec.path}: {e}")

    report = {
        "dry_run":        dry_run,
        "target_folder":  target_folder,
        "total_groups":   len(groups),
        "manual_decided": manual_decided,
        "auto_decided":   auto_decided,
        "files_moved":    len(moved),
        "files_deleted":  len(moved),
        "errors":         len(errors),
        "moved_files":    moved,
        "error_details":  errors,
        "timestamp":      _dt.now().isoformat(),
    }

    verb = "Dry-run: would move" if dry_run else "Moved"
    return jsonify({
        "success": True,
        "dry_run": dry_run,
        "report":  report,
        "message": f"{verb} {len(moved)} duplicate photo(s) to {target_folder}"
                   + (f", {len(errors)} error(s)" if errors else ""),
    })


# ─── Document deduplication endpoints ────────────────────────────────────────

@app.route("/api/scan_docs", methods=["POST"])
def api_scan_docs():
    """Start a document duplicate scan in the given folder."""
    data   = request.json or {}
    folder = data.get("folder", "").strip()

    if not folder or not Path(folder).is_dir():
        return jsonify({"error": "Invalid folder path"}), 400

    if _doc_state["running"]:
        return jsonify({"error": "Document scan already in progress"}), 409

    _doc_state["folder"]  = folder
    _doc_state["running"] = True
    _doc_state["groups"]  = []
    _doc_state["singletons"] = []

    def run_doc_pipeline():
        def progress_cb(msg: str, pct: int, scanned: int = None, total: int = None):
            payload = {"message": msg, "percent": pct, "mode": "docs"}
            if scanned is not None and total is not None:
                payload["scanned"] = scanned
                payload["total"]   = total
            socketio.emit("doc_progress", payload)

        try:
            scanner = DocumentScanner(folder)
            records = scanner.scan(progress_cb=progress_cb)

            progress_cb("Grouping duplicate documents…", 88)
            grouper         = DocumentDuplicateGrouper(records)
            groups, singles = grouper.group()

            _doc_state["groups"]      = groups
            _doc_state["singletons"]  = singles
            _doc_state["total_docs"]  = len(records)
            _doc_state["total_groups"] = len(groups)
            _doc_state["running"]     = False

            socketio.emit("doc_scan_complete", {
                "total_docs":   len(records),
                "total_groups": len(groups),
            })
        except Exception as e:
            logger.exception("Document pipeline error")
            _doc_state["running"] = False
            socketio.emit("doc_scan_error", {"error": str(e)})

    threading.Thread(target=run_doc_pipeline, daemon=True).start()
    return jsonify({"status": "started", "folder": folder})


@app.route("/api/doc_groups", methods=["GET"])
def api_doc_groups():
    """Return document duplicate groups, optionally filtered by tier or decided status."""
    tier    = request.args.get("tier")
    decided = request.args.get("decided")

    groups: list[DocumentDuplicateGroup] = _doc_state.get("groups", [])
    if tier:
        groups = [g for g in groups if g.tier == tier]
    if decided == "false":
        groups = [g for g in groups if not g.decided]
    elif decided == "true":
        groups = [g for g in groups if g.decided]

    out = [_doc_group_to_dict(g, i) for i, g in enumerate(groups)]
    return jsonify({
        "groups":    out,
        "total":     len(out),
        "undecided": sum(1 for g in _doc_state.get("groups", []) if not g.decided),
    })


@app.route("/api/doc_decide", methods=["POST"])
def api_doc_decide():
    """Record which document to keep in a duplicate group."""
    data     = request.json or {}
    group_id = data.get("group_id", "")
    kept_idx = data.get("kept_idx")

    if kept_idx is None:
        return jsonify({"error": "kept_idx required"}), 400

    grp = next((g for g in _doc_state.get("groups", []) if g.group_id == group_id), None)
    if not grp:
        return jsonify({"error": "Group not found"}), 404

    if not (0 <= kept_idx < len(grp.docs)):
        return jsonify({"error": "Invalid kept_idx"}), 400

    grp.kept_idx  = kept_idx
    grp.kept_path = grp.docs[kept_idx].path
    grp.decided   = True

    undecided = sum(1 for g in _doc_state.get("groups", []) if not g.decided)
    return jsonify({"ok": True, "undecided_remaining": undecided})


@app.route("/api/doc_status", methods=["GET"])
def api_doc_status():
    """Return the current document scan status."""
    return jsonify({
        "running":      _doc_state["running"],
        "folder":       _doc_state["folder"],
        "has_result":   bool(_doc_state["groups"] or _doc_state["singletons"]),
        "total_docs":   _doc_state.get("total_docs", 0),
        "total_groups": _doc_state.get("total_groups", 0),
        "undecided":    sum(1 for g in _doc_state.get("groups", []) if not g.decided),
    })


def _auto_select_keeper(grp: "DocumentDuplicateGroup") -> int:
    """
    Pick which document to keep in a group when no manual decision was made.
    Priority: most words → most pages → largest file → first alphabetically.
    Returns the index of the file to keep.
    """
    best_idx = 0
    best = grp.docs[0]
    for i, rec in enumerate(grp.docs[1:], 1):
        if rec.word_count > best.word_count:
            best_idx, best = i, rec
        elif rec.word_count == best.word_count:
            if rec.page_count > best.page_count:
                best_idx, best = i, rec
            elif rec.page_count == best.page_count:
                if rec.size_bytes > best.size_bytes:
                    best_idx, best = i, rec
    return best_idx


@app.route("/api/apply_doc_decisions", methods=["POST"])
def api_apply_doc_decisions():
    """
    Copy duplicate documents to target folder, then delete them from source.
    For groups without a manual decision the best file is chosen automatically
    (most words → most pages → largest size → first alphabetically).
    """
    import shutil
    from datetime import datetime as _dt

    data          = request.json or {}
    target_folder = (data.get("target_folder") or "").strip()
    dry_run       = data.get("dry_run", False)

    groups = _doc_state.get("groups", [])
    folder = _doc_state.get("folder", "")

    if not groups:
        return jsonify({"error": "No document scan results — run a scan first"}), 404

    # Default target folder: source_folder_duplicates
    if not target_folder:
        if folder:
            target_folder = str(Path(folder).parent / (Path(folder).name + "_duplicates"))
        else:
            return jsonify({"error": "Could not determine target folder — please enter one"}), 400

    target_path  = Path(target_folder)
    moved:  list[dict] = []
    errors: list[dict] = []
    auto_decided = 0
    manual_decided = 0

    for grp in groups:
        # Use manual choice if available, otherwise auto-select
        if grp.decided and grp.kept_idx is not None:
            kept_idx = grp.kept_idx
            manual_decided += 1
        else:
            kept_idx = _auto_select_keeper(grp)
            auto_decided += 1

        duplicates = [rec for i, rec in enumerate(grp.docs) if i != kept_idx]
        for rec in duplicates:
            src = Path(rec.path)
            if not src.exists():
                continue
            # Collision-safe destination name
            dest = target_path / src.name
            counter = 1
            while dest.exists() or any(m["to"] == str(dest) for m in moved):
                dest = target_path / f"{src.stem}_{counter}{src.suffix}"
                counter += 1
            try:
                if not dry_run:
                    target_path.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(dest))
                    src.unlink()
                moved.append({
                    "from":     str(src),
                    "to":       str(dest),
                    "auto":     grp.kept_idx is None,
                    "kept":     grp.docs[kept_idx].filename,
                })
            except Exception as e:
                errors.append({"file": str(rec.path), "error": str(e)})
                logger.warning(f"Failed to move {rec.path}: {e}")

    report = {
        "dry_run":        dry_run,
        "target_folder":  target_folder,
        "total_groups":   len(groups),
        "manual_decided": manual_decided,
        "auto_decided":   auto_decided,
        "files_moved":    len(moved),
        "files_deleted":  len(moved),
        "errors":         len(errors),
        "moved_files":    moved,
        "error_details":  errors,
        "timestamp":      _dt.now().isoformat(),
    }

    verb = "Dry-run: would move" if dry_run else "Moved"
    return jsonify({
        "success": True,
        "dry_run": dry_run,
        "report":  report,
        "message": f"{verb} {len(moved)} duplicate file(s) to {target_folder}"
                   + (f", {len(errors)} error(s)" if errors else ""),
    })


def _doc_safe_str(val) -> str | None:
    """Return a plain str (or None) — guards against pypdf IndirectObject values."""
    if val is None:
        return None
    try:
        s = str(val).strip()
        return s if s else None
    except Exception:
        return None


def _doc_group_to_dict(grp: DocumentDuplicateGroup, idx: int) -> dict:
    docs = []
    for i, rec in enumerate(grp.docs):
        docs.append({
            "index":       i,
            "path":        rec.path,
            "filename":    rec.filename,
            "size_bytes":  rec.size_bytes,
            "doc_type":    rec.doc_type,
            "page_count":  rec.page_count,
            "word_count":  rec.word_count,
            "title":       _doc_safe_str(rec.title),
            "author":      _doc_safe_str(rec.author),
            "created_at":  rec.created_at.isoformat()  if rec.created_at  else None,
            "modified_at": rec.modified_at.isoformat() if rec.modified_at else None,
        })

    return {
        "group_id":   grp.group_id,
        "index":      idx,
        "tier":       grp.tier,
        "similarity": float(grp.similarity),
        "doc_count":  len(grp.docs),
        "docs":       docs,
        "decided":    grp.decided,
        "kept_idx":   grp.kept_idx,
    }


@app.route("/")
def serve_index():
    """Serve the main UI."""
    return send_file(str(Path(__file__).parent / "index.html"))


# ─── SocketIO ─────────────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    status = _state.copy()
    status.pop("pipeline", None)
    status.pop("result", None)
    sio_emit("status", {"connected": True})


# ─── Entry point ──────────────────────────────────────────────────────────────
def load_most_recent_scan():
    """
    Auto-load the most recent valid scan on startup.
    Tries each saved scan newest-first and skips corrupt / incomplete ones.
    """
    scans = _list_saved_scans()
    for scan_meta in scans:
        scan_id = scan_meta.get("scan_id")
        if not scan_id:
            continue
        try:
            loaded = _load_scan_results(scan_id)
            if not loaded:
                logger.info(f"Skipping scan {scan_id} (incomplete or corrupt)")
                continue

            folder, result = loaded
            _state["folder"]            = folder
            _state["result"]            = result
            _state["scan_id"]           = scan_id
            _state["current_group_idx"] = 0

            from pipeline import Pipeline
            pipe = Pipeline(folder)
            pipe.pipeline_result_analyses_ref = result.analyses
            _state["pipeline"] = pipe

            logger.info(f"Auto-loaded scan: {scan_id} ({result.total_groups} groups, {result.total_images} images)")
            return True
        except Exception as e:
            logger.warning(f"Could not load scan {scan_id}: {e} — trying next")
            continue

    logger.info("No valid saved scan found to auto-load")
    return False


def run_server():
    print(f"\nPhotoMind starting at http://{UI_HOST}:{UI_PORT}")
    print("Loading previous scan in background — UI is available immediately\n")

    # Load the most recent scan in a background thread so Flask starts instantly
    threading.Thread(target=load_most_recent_scan, daemon=True).start()

    socketio.run(app, host=UI_HOST, port=UI_PORT, debug=False,
                 allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    run_server()
