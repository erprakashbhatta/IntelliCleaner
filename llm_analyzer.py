"""
models/llm_analyzer.py — LLM Decision Engine
Uses a vision-capable LLM (GPT-4o, Claude, or local LLaVA) augmented with
RAG-retrieved past decisions to rank duplicate photo groups and explain why.

Falls back to a rule-based scorer if no LLM API key is configured.
"""
import base64
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LLM_PROVIDER, LLM_MODEL, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    OLLAMA_BASE_URL, RAG_TOP_K_MEMORIES
)
from scanner  import ImageRecord
from grouper  import DuplicateGroup
from rag_store import RAGMemoryStore

logger = logging.getLogger(__name__)


# ─── Analysis result ──────────────────────────────────────────────────────────
class AnalysisResult:
    def __init__(self):
        self.ranking:         list[int]  = []     # indices into group.images, best first
        self.best_idx:        int        = 0
        self.confidence:      float      = 0.0
        self.reasons:         dict       = {}     # path → reason string
        self.disqualifiers:   dict       = {}     # path → list of issues
        self.recommendation:  str        = ""
        self.rag_context_used: int       = 0      # how many past decisions retrieved
        self.backend:         str        = ""


# ─── Analyzer ─────────────────────────────────────────────────────────────────
class LLMAnalyzer:
    """
    Rank images within a DuplicateGroup using LLM vision analysis + RAG memory.

    Usage:
        analyzer = LLMAnalyzer(memory_store)
        result   = analyzer.analyze(group)
        group.best_guess_idx = result.best_idx
    """

    def __init__(self, memory: RAGMemoryStore):
        self.memory   = memory
        self._client  = None
        self._backend = self._init_backend()

    # ── Public ────────────────────────────────────────────────────────────────
    def analyze(self, group: DuplicateGroup) -> AnalysisResult:
        if self._backend == "none" or not group.images:
            return self._fallback_analysis(group)

        # 1. Retrieve similar past decisions from RAG
        group_emb = self._mean_embedding(group.images)
        memories  = self.memory.retrieve_similar(group_emb, RAG_TOP_K_MEMORIES)

        # 2. Build prompt
        system_prompt = self._build_system_prompt(memories)
        user_content  = self._build_user_content(group)

        # 3. Call LLM
        try:
            raw = self._call_llm(system_prompt, user_content)
            return self._parse_response(raw, group, len(memories))
        except Exception as e:
            logger.warning(f"LLM call failed ({e}), using fallback scorer")
            return self._fallback_analysis(group)

    # ── Backend init ──────────────────────────────────────────────────────────
    def _init_backend(self) -> str:
        if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=OPENAI_API_KEY)
                logger.info(f"LLM backend: OpenAI ({LLM_MODEL})")
                return "openai"
            except ImportError:
                pass

        if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                logger.info(f"LLM backend: Anthropic ({LLM_MODEL})")
                return "anthropic"
            except ImportError:
                pass

        if LLM_PROVIDER == "ollama":
            try:
                import ollama as ol
                ol.list()   # connectivity check
                self._client = ol
                logger.info(f"LLM backend: Ollama ({LLM_MODEL})")
                return "ollama"
            except Exception:
                pass

        logger.warning("No LLM backend available — using rule-based scoring only")
        return "none"

    # ── Prompt builders ───────────────────────────────────────────────────────
    def _build_system_prompt(self, memories: list[dict]) -> str:
        mem_text = ""
        if memories:
            mem_text = "\n\nPAST DECISIONS (use as few-shot examples):\n"
            for i, m in enumerate(memories[:6], 1):
                acc = "✓ Human accepted LLM suggestion" if m.get("human_accepted") else "✗ Human overrode LLM — learn from this"
                tags = m.get("reason_tags", "")
                if isinstance(tags, list):
                    tags = ", ".join(tags)
                mem_text += (
                    f"  {i}. Group type={m.get('tier','?')}, "
                    f"{m.get('image_count','?')} photos. "
                    f"Kept: {Path(m.get('kept_path','')).name}. "
                    f"Tags: {tags or 'none'}. {acc}.\n"
                )

        return f"""You are an expert photo quality analyst helping select the best photo from near-duplicate groups.

Your job is to rank photos from best to worst and explain each disqualifier clearly.

EVALUATION CRITERIA (in priority order):
1. All faces must have eyes open (most important — closed eyes = disqualify)
2. All subjects must be looking toward the camera (not looking away)
3. Image must be sharp / in focus (not blurry, not motion-blurred)
4. Good exposure (not overexposed/underexposed)
5. Natural expressions / smiles preferred
6. Clean background preferred
7. Good composition (rule of thirds, subject placement)
8. No bad hand gestures or awkward poses

DISQUALIFIER TAGS (use these exact strings in your response):
eyes_closed, looking_away, blurry, bad_exposure, bad_smile,
hand_gesture_bad, background_unclear, motion_blur, face_occluded, bad_composition
{mem_text}
RESPONSE FORMAT: Respond ONLY with valid JSON. No markdown, no explanation outside the JSON.
{{
  "ranking": [0, 2, 1],           // indices of images, best first
  "best_idx": 0,                  // index of the single best image
  "confidence": 0.87,             // 0.0–1.0 how confident you are
  "reasons": {{
    "0": "All faces forward, eyes open, sharp, well-exposed",
    "1": "Person on left has eyes closed",
    "2": "Slightly blurry, one person looking away"
  }},
  "disqualifiers": {{
    "0": [],
    "1": ["eyes_closed"],
    "2": ["blurry", "looking_away"]
  }},
  "recommendation": "Keep photo 0 — clearest face quality. Delete 1 (eyes closed), 2 (blur + gaze)."
}}"""

    def _build_user_content(self, group: DuplicateGroup) -> list:
        """Build multimodal message content with images + quality metadata."""
        parts = [{
            "type": "text",
            "text": (
                f"Please analyze this group of {len(group.images)} near-duplicate photos "
                f"(similarity tier: {group.tier}, similarity: {group.similarity:.2f}).\n\n"
                "Quality scores from automatic analysis:\n" +
                "\n".join(
                    f"  Photo {i}: {self._fmt_quality(r)}"
                    for i, r in enumerate(group.images)
                ) + "\n\nImages:"
            )
        }]

        for i, rec in enumerate(group.images):
            parts.append({"type": "text", "text": f"\nPhoto {i} — {rec.filename}:"})
            b64 = self._image_to_b64(rec.path)
            if b64:
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
                })
            else:
                parts.append({"type": "text", "text": "[image could not be loaded]"})

        return parts

    # ── LLM call ──────────────────────────────────────────────────────────────
    def _call_llm(self, system: str, user_content: list) -> str:
        if self._backend == "openai":
            resp = self._client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_content},
                ],
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content

        if self._backend == "anthropic":
            # Convert OpenAI-style content to Anthropic format
            ant_content = []
            for part in user_content:
                if part["type"] == "text":
                    ant_content.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    b64 = url.split(",", 1)[1]
                    ant_content.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}
                    })
            resp = self._client.messages.create(
                model=LLM_MODEL,
                max_tokens=800,
                system=system,
                messages=[{"role": "user", "content": ant_content}],
            )
            return resp.content[0].text

        if self._backend == "ollama":
            # Ollama: extract first image for vision models
            images = []
            text_parts = []
            for part in user_content:
                if part["type"] == "image_url":
                    b64 = part["image_url"]["url"].split(",", 1)[1]
                    images.append(b64)
                elif part["type"] == "text":
                    text_parts.append(part["text"])
            resp = self._client.chat(
                model=LLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": system + "\n\n" + "\n".join(text_parts),
                    "images": images[:4],   # most Ollama vision models: max 4 images
                }]
            )
            return resp["message"]["content"]

        raise RuntimeError("No backend")

    # ── Response parser ───────────────────────────────────────────────────────
    def _parse_response(
        self, raw: str, group: DuplicateGroup, n_memories: int
    ) -> AnalysisResult:
        result = AnalysisResult()
        result.backend          = self._backend
        result.rag_context_used = n_memories
        try:
            data = json.loads(raw.strip())
            n    = len(group.images)

            result.ranking       = [int(i) for i in data.get("ranking", list(range(n))) if 0 <= int(i) < n]
            result.best_idx      = int(data.get("best_idx", result.ranking[0] if result.ranking else 0))
            result.confidence    = float(data.get("confidence", 0.7))
            result.reasons       = {str(k): str(v) for k, v in data.get("reasons", {}).items()}
            result.disqualifiers = {str(k): v for k, v in data.get("disqualifiers", {}).items()}
            result.recommendation = str(data.get("recommendation", ""))
        except Exception as e:
            logger.warning(f"LLM response parse error: {e}\nRaw: {raw[:300]}")
            return self._fallback_analysis(group)

        return result

    # ── Fallback rule-based scorer ────────────────────────────────────────────
    def _fallback_analysis(self, group: DuplicateGroup) -> AnalysisResult:
        result = AnalysisResult()
        result.backend = "rule_based"

        scored = []
        for i, rec in enumerate(group.images):
            score = rec.quality_score if rec.quality_score is not None else 0.5
            scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        result.ranking    = [i for i, _ in scored]
        result.best_idx   = result.ranking[0] if result.ranking else 0
        result.confidence = 0.60
        result.reasons    = {
            str(i): f"Quality score: {score:.3f}" for i, score in scored
        }
        result.recommendation = (
            f"Rule-based selection: Photo {result.best_idx} has highest quality score "
            f"({group.images[result.best_idx].quality_score:.3f}). "
            "Configure LLM_PROVIDER for smarter analysis."
        )
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _mean_embedding(records: list[ImageRecord]) -> Optional[list[float]]:
        embs = [r.embedding for r in records if r.embedding]
        if not embs:
            return None
        arr = np.mean(np.array(embs, dtype=np.float32), axis=0)
        return arr.tolist()

    @staticmethod
    def _image_to_b64(path: str, max_dim: int = 768) -> Optional[str]:
        try:
            from PIL import Image
            import io
            img = Image.open(path).convert("RGB")
            # Downscale for API efficiency
            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=75)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return None

    @staticmethod
    def _fmt_quality(rec: ImageRecord) -> str:
        d = rec.quality_details or {}
        parts = [f"score={rec.quality_score:.2f}" if rec.quality_score else "score=?"]
        if d.get("face_count"):
            parts.append(f"faces={d['face_count']}")
        if "sharpness" in d:
            parts.append(f"sharp={d['sharpness']:.2f}")
        if "exposure" in d:
            parts.append(f"exposure={d['exposure']:.2f}")
        if d.get("issues"):
            parts.append(f"issues=[{', '.join(d['issues'])}]")
        return "  ".join(parts)
