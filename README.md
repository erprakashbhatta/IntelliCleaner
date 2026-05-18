# PhotoMind — AI Photo Deduplication with Continuous Learning

A self-learning, LLM-powered photo deduplication system that finds near-duplicate photos,
scores their quality, asks for your input, and learns from every decision you make.

---

## Features

- **3-tier duplicate detection**: exact hash → perceptual hash → CLIP semantic similarity + burst detection
- **AI quality scoring**: sharpness, face quality, eye openness, gaze direction, smile, exposure, composition
- **LLM vision analysis**: GPT-4o / Claude / local LLaVA ranks photos and explains every disqualifier
- **RAG memory**: every decision stored in ChromaDB — future groups get relevant past decisions as context
- **Continuous learning**: overridden AI suggestions are flagged and learned from
- **Beautiful web UI**: side-by-side photo review with real-time progress
- **CLI mode**: headless scanning with auto-decision support

---

## Quick Start

### 1. Install dependencies

```bash
cd photo_dedup
pip install -r requirements.txt
```

For HEIC (iPhone photos):
```bash
pip install pillow-heif
```

For RAW files (Canon, Nikon, Sony):
```bash
pip install rawpy
```

For local CLIP embeddings:
```bash
pip install openai-clip
# or
pip install git+https://github.com/openai/CLIP.git
```

### 2. Configure your LLM

```bash
cp .env.example .env
# Edit .env and add your API key
```

**Option A — OpenAI (best results):**
```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```

**Option B — Anthropic Claude:**
```
LLM_PROVIDER=anthropic
LLM_MODEL=claude-opus-4-6
ANTHROPIC_API_KEY=sk-ant-...
```

**Option C — Local Ollama (free, private):**
```bash
# Install ollama: https://ollama.ai
ollama pull llava:13b
```
```
LLM_PROVIDER=ollama
LLM_MODEL=llava:13b
```

### 3. Run

**Web UI (recommended):**
```bash
python main.py
# Opens http://127.0.0.1:5050 in your browser
```

**CLI (headless):**
```bash
# Scan and print report
python main.py --folder /path/to/photos --report

# Auto-decide high-confidence duplicates
python main.py --folder /path/to/photos --output /path/to/output --auto-decide

# Dry run first (preview only)
python main.py --folder /path/to/photos --output /path/to/output --auto-decide --dry-run
```

---

## How It Works

```
1. SCAN           Walk directory → load images (JPEG/PNG/HEIC/RAW)
                  Extract EXIF metadata, timestamps, GPS, camera model

2. EMBED          CLIP ViT-B/32 → 512-dim visual embeddings per image

3. SCORE          OpenCV + MediaPipe → sharpness, face quality, eye openness,
                  gaze direction, smile score, exposure, composition

4. GROUP          Three-tier clustering:
                  ① SHA-256 exact hash
                  ② pHash Hamming distance (≤12 = near duplicate)
                  ③ CLIP cosine similarity (≥0.88 = same scene)
                  ④ Burst detection (same camera, ≤5 seconds apart)

5. ANALYZE        For each group: retrieve similar past decisions from ChromaDB
                  (RAG) → send images + quality scores + memory to LLM
                  → get ranked list + disqualifier tags + confidence score

6. REVIEW         Web UI shows side-by-side comparison with AI suggestion.
                  Human selects best photo, tags reason, adds notes.

7. LEARN          Decision saved to SQLite + ChromaDB.
                  If human overrode AI → flagged as negative signal.
                  Next similar group retrieves this decision as context.
```

---

## Project Structure

```
photo_dedup/
├── main.py              # Entry point (web UI + CLI)
├── pipeline.py          # Main orchestration
├── config.py            # All configuration & thresholds
├── requirements.txt
├── .env.example
│
├── core/
│   ├── scanner.py       # Image loading, EXIF, thumbnails, hashing
│   ├── embedder.py      # CLIP / ResNet visual embeddings
│   ├── quality.py       # OpenCV + MediaPipe quality scoring
│   └── grouper.py       # 3-tier duplicate clustering
│
├── models/
│   └── llm_analyzer.py  # LLM analysis with RAG context
│
├── memory/
│   └── rag_store.py     # SQLite + ChromaDB decision memory
│
├── ui/
│   ├── server.py        # Flask + SocketIO API server
│   └── static/
│       └── index.html   # Web review UI
│
└── data/                # Auto-created
    ├── decisions.db      # SQLite decision history
    ├── chromadb/         # Vector search index
    └── thumbnails/       # Cached image thumbnails
```

---

## Configuration (config.py)

| Setting | Default | Description |
|---|---|---|
| `PHASH_NEAR_THRESHOLD` | 12 | Hamming distance for near-duplicate |
| `CLIP_SEMANTIC_THRESHOLD` | 0.88 | Cosine similarity for same-scene |
| `BURST_TIME_WINDOW_SEC` | 5 | Seconds apart to consider burst |
| `LLM_CONFIDENCE_AUTO_THRESHOLD` | 0.92 | Auto-decide above this confidence |
| `RAG_TOP_K_MEMORIES` | 8 | Past decisions retrieved per group |
| `QUALITY_WEIGHTS` | see config | Weight of each quality dimension |

---

## Output Folder Structure

After decisions are applied:
```
/your_output_folder/
├── best_photo_001.jpg    ← kept (best quality)
├── best_photo_002.jpg
├── ...
└── review/               ← moved here (not deleted!)
    ├── duplicate_001.jpg
    └── duplicate_002.jpg
```

**Files are never deleted** — duplicates are moved to `/review/`.
You can delete the review folder once you're satisfied.

---

## Privacy Note

If using OpenAI or Anthropic, thumbnail versions of your photos (768px)
are sent to the API for analysis. Use **Ollama (local)** mode if privacy
is a concern for personal/family photos.
