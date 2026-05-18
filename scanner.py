"""
core/scanner.py — Image Scanner
Walks a local directory, loads images, extracts EXIF metadata,
generates perceptual hashes, and prepares images for embedding.
"""
import os
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Iterator, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import imagehash
from PIL import Image, ExifTags
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False

try:
    import rawpy
    RAW_SUPPORTED = True
except ImportError:
    RAW_SUPPORTED = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (SUPPORTED_EXTENSIONS, THUMBNAIL_SIZE, THUMBNAILS_DIR,
                    MAX_IMAGE_SIZE_MB, NUM_WORKERS, SCAN_EXCLUDED_DIRS)
from scan_cache import get_cache

# Increase PIL's max image pixels to handle large images (prevents DecompressionBombWarning)
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 200000000  # Allow up to 200 million pixels

console = Console()
logger  = logging.getLogger(__name__)


# ─── Data model ───────────────────────────────────────────────────────────────
@dataclass
class ImageRecord:
    """All extracted info about one image file."""
    path:       str
    filename:   str
    file_hash:  str              # SHA-256 — exact duplicate detection
    phash:      str              # Perceptual hash (imagehash)
    size_bytes: int
    width:      int
    height:     int
    mode:       str              # RGB, RGBA, L …
    created_at: Optional[datetime] = None
    taken_at:   Optional[datetime] = None   # EXIF DateTimeOriginal
    camera:     Optional[str] = None
    gps_lat:    Optional[float] = None
    gps_lon:    Optional[float] = None
    thumbnail_path: Optional[str] = None
    embedding:  Optional[list] = None       # CLIP — filled later
    quality_score: Optional[float] = None   # filled by QualityScorer
    quality_details: dict = field(default_factory=dict)
    error:      Optional[str] = None


# ─── Scanner ──────────────────────────────────────────────────────────────────
class ImageScanner:
    """
    Recursively scan a directory and return ImageRecord objects.

    Usage:
        scanner = ImageScanner("/path/to/photos")
        records = list(scanner.scan())
    """

    def __init__(self, root: str, recursive: bool = True, create_thumbnails: bool = False,
                 excluded_dirs: Optional[set] = None):
        self.root              = Path(root)
        self.recursive         = recursive
        self.create_thumbnails = create_thumbnails
        # Merge caller-supplied exclusions with global config
        self.excluded_dirs: set[str] = (SCAN_EXCLUDED_DIRS | (excluded_dirs or set()))
        self._preloaded: dict = {}   # filled by scan() before workers start

    # ── Public API ────────────────────────────────────────────────────────────
    def scan(self, progress_cb: Optional[Callable[..., None]] = None) -> list["ImageRecord"]:
        """Scan directory and return list of ImageRecords."""
        paths = self._collect_paths()
        total = len(paths)
        console.print(f"[bold cyan]Found {total} image files in {self.root}[/]")

        # ── Bulk cache preload (1 SQL query instead of N) ─────────────────────
        cache           = get_cache()
        self._preloaded = cache.preload_folder(str(self.root))
        hits            = sum(1 for p in paths
                              if self._make_cache_key(p) in self._preloaded)
        console.print(f"[dim]Cache: {hits}/{total} files already cached[/]")

        records: list[ImageRecord] = []
        completed = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Scanning images...", total=total)
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {executor.submit(self._process_file, p): p for p in paths}
                for future in as_completed(futures):
                    rec = future.result()
                    if rec is not None:
                        records.append(rec)
                    completed += 1
                    progress.advance(task)
                    if progress_cb:
                        progress_cb(
                            f"Scanning images... ({completed}/{total})",
                            5 + int(15 * completed / max(total, 1)),
                            completed,
                            total,
                        )

        ok  = sum(1 for r in records if r.error is None)
        err = sum(1 for r in records if r.error)
        console.print(f"[green]Loaded {ok} images[/]  [red]{err} errors[/]")
        return records

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _is_excluded(self, path: Path) -> bool:
        """Return True if any component of path is in the exclusion set."""
        return any(part in self.excluded_dirs for part in path.parts)

    def _make_cache_key(self, path: Path):
        """Cache lookup key — call stat() once and reuse."""
        try:
            s = path.stat()
            return (str(path), s.st_size, round(s.st_mtime, 3))
        except OSError:
            return None

    def _collect_paths(self) -> list[Path]:
        """Collect eligible image paths, skipping excluded directories."""
        max_bytes = MAX_IMAGE_SIZE_MB * 1024 * 1024
        pattern   = "**/*" if self.recursive else "*"
        result    = []
        for p in self.root.glob(pattern):
            if not p.is_file():
                continue
            if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if self._is_excluded(p):
                continue
            # Only call stat() once here; cache key reuses it in _process_file
            try:
                if p.stat().st_size <= max_bytes:
                    result.append(p)
            except OSError:
                continue
        return result

    def _process_file(self, path: Path) -> Optional["ImageRecord"]:
        try:
            stat      = path.stat()
            # ── O(1) lookup in preloaded dict (no SQL round-trip) ─────────────
            cache_key = (str(path), stat.st_size, round(stat.st_mtime, 3))
            cached    = self._preloaded.get(cache_key)

            # ── Cache hit: all scan fields already known ───────────────────────
            if cached and cached.get("file_hash") and cached.get("phash") and cached.get("width"):
                taken_at = None
                if cached.get("taken_at"):
                    try:
                        taken_at = datetime.fromisoformat(cached["taken_at"])
                    except ValueError:
                        pass
                rec = ImageRecord(
                    path=str(path),
                    filename=path.name,
                    file_hash=cached["file_hash"],
                    phash=cached["phash"],
                    size_bytes=stat.st_size,
                    width=cached.get("width", 0),
                    height=cached.get("height", 0),
                    mode=cached.get("img_mode", ""),
                    created_at=datetime.fromtimestamp(stat.st_ctime),
                    taken_at=taken_at,
                    camera=cached.get("camera"),
                    gps_lat=cached.get("gps_lat"),
                    gps_lon=cached.get("gps_lon"),
                )
                if cached.get("embedding"):
                    rec.embedding = cached["embedding"]
                if cached.get("quality_score") is not None:
                    rec.quality_score   = cached["quality_score"]
                    rec.quality_details = cached.get("quality_details") or {}
                return rec

            # ── Cache miss: process from disk ──────────────────────────────────
            img = self._open_image(path)
            if img is None:
                return None

            # Palette images with byte transparency must go via RGBA to avoid PIL warning
            if img.mode == "P" and "transparency" in img.info:
                img = img.convert("RGBA")
            img_rgb = img.convert("RGB")

            ph        = str(imagehash.phash(img_rgb))
            file_hash = self._file_hash(path)
            exif      = self._extract_exif(img)
            thumb_path = self._save_thumbnail(img_rgb, path) if self.create_thumbnails else None

            # Persist to cache (DB write — thread-safe via WAL)
            get_cache().put(str(path), stat.st_size, stat.st_mtime, {
                "file_hash":  file_hash,
                "phash":      ph,
                "width":      img.width,
                "height":     img.height,
                "img_mode":   img.mode,
                "taken_at":   exif.get("taken_at").isoformat() if exif.get("taken_at") else None,
                "camera":     exif.get("camera"),
                "gps_lat":    exif.get("gps_lat"),
                "gps_lon":    exif.get("gps_lon"),
            })

            return ImageRecord(
                path=str(path),
                filename=path.name,
                file_hash=file_hash,
                phash=ph,
                size_bytes=stat.st_size,
                width=img.width,
                height=img.height,
                mode=img.mode,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                taken_at=exif.get("taken_at"),
                camera=exif.get("camera"),
                gps_lat=exif.get("gps_lat"),
                gps_lon=exif.get("gps_lon"),
                thumbnail_path=str(thumb_path) if thumb_path else None,
            )
        except Exception as e:
            logger.warning(f"Error processing {path}: {e}")
            return ImageRecord(
                path=str(path),
                filename=path.name,
                file_hash="", phash="",
                size_bytes=0, width=0, height=0, mode="",
                error=str(e),
            )

    def _open_image(self, path: Path) -> Optional[Image.Image]:
        suffix = path.suffix.lower()
        if suffix in {".raw", ".cr2", ".nef", ".arw"} and RAW_SUPPORTED:
            import rawpy, numpy as np
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess()
            return Image.fromarray(rgb)
        return Image.open(path)

    @staticmethod
    def _file_hash(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(chunk))
        return h.hexdigest()

    @staticmethod
    def _extract_exif(img: Image.Image) -> dict:
        result = {}
        try:
            raw_exif = img._getexif()
            if not raw_exif:
                return result
            exif = {ExifTags.TAGS.get(k, k): v for k, v in raw_exif.items()}

            # DateTime
            for tag in ("DateTimeOriginal", "DateTime"):
                if tag in exif:
                    try:
                        result["taken_at"] = datetime.strptime(exif[tag], "%Y:%m:%d %H:%M:%S")
                        break
                    except ValueError:
                        pass

            # Camera
            make  = exif.get("Make", "")
            model = exif.get("Model", "")
            if make or model:
                result["camera"] = f"{make} {model}".strip()

            # GPS
            gps_info = exif.get("GPSInfo")
            if gps_info:
                gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
                def _dms(vals, ref):
                    d, m, s = vals
                    dec = float(d) + float(m) / 60 + float(s) / 3600
                    return -dec if ref in ("S", "W") else dec
                if "GPSLatitude" in gps and "GPSLongitude" in gps:
                    result["gps_lat"] = _dms(gps["GPSLatitude"],  gps.get("GPSLatitudeRef",  "N"))
                    result["gps_lon"] = _dms(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
        except Exception:
            pass
        return result

    @staticmethod
    def _save_thumbnail(img: Image.Image, orig_path: Path) -> Optional[Path]:
        try:
            thumb = img.copy()
            thumb.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
            thumb_path = THUMBNAILS_DIR / f"{orig_path.stem}_{orig_path.stat().st_size}.jpg"
            thumb.save(thumb_path, "JPEG", quality=85)
            return thumb_path
        except Exception:
            return None
