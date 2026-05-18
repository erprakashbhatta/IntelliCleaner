"""
core/quality.py — Photo Quality Scorer
Scores each image on: sharpness, face quality (open eyes, forward gaze,
smile), exposure, and composition. Returns a composite 0–100 score plus
per-dimension details used by the LLM for explanation.
"""
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import QUALITY_WEIGHTS
from scanner import ImageRecord

logger = logging.getLogger(__name__)


# ── Optional deps — graceful degradation ──────────────────────────────────────
try:
    import mediapipe as mp
    _mp_face  = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    _mp_mesh  = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=6,
        refine_landmarks=True, min_detection_confidence=0.5
    )
    _mp_pose  = mp.solutions.holistic.Holistic(static_image_mode=True)
    MEDIAPIPE_OK = True
except Exception:
    MEDIAPIPE_OK = False
    logger.warning("MediaPipe not available — face quality scoring disabled")

try:
    import face_recognition as fr
    FACE_RECOG_OK = True
except Exception:
    FACE_RECOG_OK = False


# ─── Quality result ───────────────────────────────────────────────────────────
class QualityResult:
    def __init__(self):
        self.sharpness:     float = 0.0   # 0–1
        self.exposure:      float = 0.0
        self.face_count:    int   = 0
        self.faces_open_eyes: float = 0.0  # fraction of detected faces with eyes open
        self.faces_forward:   float = 0.0  # fraction looking at camera
        self.smile_score:     float = 0.0
        self.composition:     float = 0.0
        self.composite:       float = 0.0
        self.issues:          list[str] = []

    def to_dict(self) -> dict:
        return {
            "sharpness":       round(self.sharpness, 3),
            "exposure":        round(self.exposure, 3),
            "face_count":      self.face_count,
            "faces_open_eyes": round(self.faces_open_eyes, 3),
            "faces_forward":   round(self.faces_forward, 3),
            "smile_score":     round(self.smile_score, 3),
            "composition":     round(self.composition, 3),
            "composite":       round(self.composite, 3),
            "issues":          self.issues,
        }


# ─── Scorer ───────────────────────────────────────────────────────────────────
class QualityScorer:
    """
    Score a list of ImageRecords. Fills rec.quality_score and rec.quality_details.

    Usage:
        scorer  = QualityScorer()
        records = scorer.score(records)
    """

    def score(self, records: list[ImageRecord]) -> list[ImageRecord]:
        from scan_cache import get_cache
        from pathlib import Path as _Path
        cache = get_cache()

        for rec in records:
            if rec.error:
                continue
            # Use cached quality if already populated (scanner cache hit)
            if rec.quality_score is not None:
                continue
            # Check cache
            try:
                stat   = _Path(rec.path).stat()
                cached = cache.get(rec.path, stat.st_size, stat.st_mtime)
                if cached and cached.get("quality_score") is not None:
                    rec.quality_score   = cached["quality_score"]
                    rec.quality_details = cached.get("quality_details") or {}
                    continue
            except OSError:
                pass
            # Compute
            try:
                result = self._score_one(rec)
                rec.quality_score   = result.composite
                rec.quality_details = result.to_dict()
                # Save to cache
                try:
                    stat = _Path(rec.path).stat()
                    cache.merge(rec.path, stat.st_size, stat.st_mtime, {
                        "quality_score":   rec.quality_score,
                        "quality_details": rec.quality_details,
                    })
                except OSError:
                    pass
            except Exception as e:
                logger.warning(f"Quality scoring failed for {rec.filename}: {e}")
                rec.quality_score   = 0.5
                rec.quality_details = {"error": str(e)}
        return records

    def score_one(self, rec: ImageRecord) -> QualityResult:
        return self._score_one(rec)

    # ── Internal ──────────────────────────────────────────────────────────────
    def _score_one(self, rec: ImageRecord) -> QualityResult:
        result = QualityResult()
        img_bgr = cv2.imread(rec.path)
        if img_bgr is None:
            result.issues.append("could_not_load")
            return result

        # ── Sharpness via Laplacian variance ──────────────────────────────────
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # 500+ is very sharp; below 50 is blurry. Normalise to 0–1
        result.sharpness = min(1.0, lap_var / 500.0)
        if result.sharpness < 0.25:
            result.issues.append("blurry")

        # ── Exposure ──────────────────────────────────────────────────────────
        result.exposure = self._score_exposure(img_bgr)
        if result.exposure < 0.3:
            result.issues.append("bad_exposure")

        # ── Composition (rule-of-thirds proxy) ────────────────────────────────
        result.composition = self._score_composition(gray)

        # ── Face quality (MediaPipe) ───────────────────────────────────────────
        if MEDIAPIPE_OK:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self._score_faces(img_rgb, result)

        # ── Composite ─────────────────────────────────────────────────────────
        w = QUALITY_WEIGHTS
        result.composite = (
            w["sharpness"]    * result.sharpness
          + w["exposure"]     * result.exposure
          + w["composition"]  * result.composition
          + w["face_quality"] * result.faces_open_eyes
          + w["smile"]        * result.smile_score
          + w["gaze_forward"] * result.faces_forward
        )
        result.composite = round(min(1.0, result.composite), 4)
        return result

    # ── Exposure ──────────────────────────────────────────────────────────────
    @staticmethod
    def _score_exposure(img_bgr: np.ndarray) -> float:
        """Penalise over/underexposure using histogram clipping fractions."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        total = hist.sum()
        if total == 0:
            return 0.5
        under = hist[:15].sum() / total    # very dark pixels
        over  = hist[240:].sum() / total   # blown-out pixels
        penalty = under + over
        return max(0.0, 1.0 - penalty * 4.0)

    # ── Composition ───────────────────────────────────────────────────────────
    @staticmethod
    def _score_composition(gray: np.ndarray) -> float:
        """
        Rule-of-thirds: detect main subject saliency near thirds intersections.
        Rough proxy: variance in thirds zones vs. centre.
        """
        h, w = gray.shape
        th, tw = h // 3, w // 3
        zones  = [
            gray[th:2*th, tw:2*tw],    # centre
            gray[0:th, 0:tw],           # top-left third
            gray[0:th, 2*tw:],          # top-right third
            gray[2*th:, 0:tw],          # bottom-left third
            gray[2*th:, 2*tw:],         # bottom-right third
        ]
        vars_ = [float(z.var()) for z in zones if z.size > 0]
        if not vars_:
            return 0.5
        max_var  = max(vars_)
        cent_var = float(zones[0].var()) if zones[0].size > 0 else 0.0
        # Score higher when subject energy is in thirds zones, not just centre
        off_centre = max(v for v in vars_[1:]) if vars_[1:] else 0.0
        score = (off_centre / (max_var + 1e-6)) * 0.5 + 0.5
        return round(min(1.0, score), 3)

    # ── Face quality ──────────────────────────────────────────────────────────
    def _score_faces(self, img_rgb: np.ndarray, result: QualityResult):
        try:
            detection = _mp_face.process(img_rgb)
            if not detection.detections:
                return

            result.face_count = len(detection.detections)

            mesh_result = _mp_mesh.process(img_rgb)
            if not mesh_result.multi_face_landmarks:
                return

            open_eyes_list, forward_list, smile_list = [], [], []
            for face_lm in mesh_result.multi_face_landmarks:
                lms = face_lm.landmark

                # ── Eye openness (EAR — eye aspect ratio) ─────────────────────
                ear = self._eye_aspect_ratio(lms)
                open_eyes_list.append(1.0 if ear > 0.20 else 0.0)
                if ear <= 0.20:
                    result.issues.append("eyes_closed")

                # ── Gaze forward (head pose via nose-tip vs face bbox) ─────────
                nose_x  = lms[1].x
                left_x  = lms[234].x
                right_x = lms[454].x
                centre  = (left_x + right_x) / 2.0
                deviation = abs(nose_x - centre)
                forward_score = max(0.0, 1.0 - deviation * 6.0)
                forward_list.append(forward_score)
                if forward_score < 0.5:
                    result.issues.append("looking_away")

                # ── Smile (mouth width ratio) ──────────────────────────────────
                mouth_left  = lms[61]
                mouth_right = lms[291]
                mouth_top   = lms[13]
                mouth_bot   = lms[14]
                width  = abs(mouth_right.x - mouth_left.x)
                height = abs(mouth_top.y  - mouth_bot.y)
                smile  = min(1.0, width / max(height + 1e-6, 0.04))
                smile_list.append(smile)

            result.faces_open_eyes = float(np.mean(open_eyes_list)) if open_eyes_list else 0.0
            result.faces_forward   = float(np.mean(forward_list))   if forward_list   else 0.0
            result.smile_score     = float(np.mean(smile_list))      if smile_list     else 0.0

        except Exception as e:
            logger.debug(f"Face scoring error: {e}")

    # ── Eye aspect ratio ──────────────────────────────────────────────────────
    @staticmethod
    def _eye_aspect_ratio(lms) -> float:
        """
        Simplified EAR using MediaPipe face mesh landmark indices.
        Left eye: 159,145 (vertical) / 133,33 (horizontal)
        Right eye: 386,374 (vertical) / 362,263 (horizontal)
        """
        try:
            def dist(a, b):
                return ((lms[a].x - lms[b].x)**2 + (lms[a].y - lms[b].y)**2) ** 0.5

            left_ear  = dist(159, 145) / (dist(133, 33)  + 1e-6)
            right_ear = dist(386, 374) / (dist(362, 263) + 1e-6)
            return (left_ear + right_ear) / 2.0
        except Exception:
            return 1.0   # assume open if can't measure
