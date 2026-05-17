"""
config.py — Central configuration for the Photo Deduplication System
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MEMORY_DIR = DATA_DIR / "memory"
CHROMA_DIR = DATA_DIR / "chromadb"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"
LOGS_DIR = DATA_DIR / "logs"
SCANS_DIR = DATA_DIR / "scans"

for d in [DATA_DIR, MEMORY_DIR, CHROMA_DIR, THUMBNAILS_DIR, LOGS_DIR, SCANS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "decisions.db"


# ─── Similarity thresholds ────────────────────────────────────────────────────
PHASH_EXACT_THRESHOLD = 0        # Hamming distance — exact perceptual match
PHASH_NEAR_THRESHOLD  = 13       # Near-duplicate (resized, recompressed) — 80% pHash similarity
CLIP_SEMANTIC_THRESHOLD = 0.80   # Cosine similarity — same scene / duplicate scene
BURST_TIME_WINDOW_SEC   = 5      # Photos within 5s → treat as burst group


# ─── Quality scoring weights ──────────────────────────────────────────────────
QUALITY_WEIGHTS = {
    "sharpness":      0.30,
    "face_quality":   0.25,
    "exposure":       0.15,
    "smile":          0.10,
    "gaze_forward":   0.10,
    "composition":    0.10,
}


# ─── LLM config ───────────────────────────────────────────────────────────────
LLM_PROVIDER        = os.getenv("LLM_PROVIDER", "openai")   # "openai" | "ollama" | "anthropic"
LLM_MODEL           = os.getenv("LLM_MODEL", "gpt-4o")       # or "llava:13b" for local
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

LLM_CONFIDENCE_AUTO_THRESHOLD = 0.92   # Auto-decide without human review above this
RAG_TOP_K_MEMORIES             = 8     # How many past decisions to retrieve for context


# ─── Processing ───────────────────────────────────────────────────────────────
THUMBNAIL_SIZE     = (400, 400)
MAX_IMAGE_SIZE_MB  = 50
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif",
    ".tiff", ".tif", ".bmp", ".webp", ".raw", ".cr2", ".nef", ".arw"
}

SUPPORTED_DOC_EXTENSIONS = {".pdf", ".docx", ".doc"}
DOC_TEXT_SIMILARITY_THRESHOLD = 0.85   # TF-IDF cosine similarity threshold for near-duplicate docs

BATCH_SIZE = 32   # Images processed per batch for embeddings
CPU_COUNT = os.cpu_count() or 4
NUM_WORKERS = min(8, max(4, CPU_COUNT))   # Conservative: too many threads thrash disk I/O

# ─── Folders auto-excluded from every scan ────────────────────────────────────
SCAN_EXCLUDED_DIRS = {
    "duplicates_backup",   # our own backup output
    ".venv", "venv", ".env",
    "__pycache__", ".git", ".svn",
    "node_modules",
    "$RECYCLE.BIN", "System Volume Information",
    "Windows", "Program Files", "Program Files (x86)",
}

# ─── Grouper limits ───────────────────────────────────────────────────────────
# Skip CLIP semantic clustering when more than this many records remain after
# exact+pHash tiers (semantic step is O(n²) in memory for large n)
MAX_SEMANTIC_CLUSTER_SIZE = 8_000


# ─── UI server ────────────────────────────────────────────────────────────────
UI_HOST = "127.0.0.1"
UI_PORT = 5050
DEBUG   = os.getenv("DEBUG", "false").lower() == "true"


# ─── Reason tags (shown in UI for human labelling) ────────────────────────────
REASON_TAGS = [
    "eyes_closed",
    "looking_away",
    "blurry",
    "bad_exposure",
    "bad_smile",
    "hand_gesture_bad",
    "background_unclear",
    "motion_blur",
    "face_occluded",
    "duplicate_exact",
    "duplicate_similar",
    "better_composition",
    "best_overall",
    "user_preference",
]
