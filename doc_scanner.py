"""
doc_scanner.py — Document Scanner
Scans for PDF and Word (.docx/.doc) files, extracts text and metadata
for duplicate detection via hash and content similarity.
"""
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

try:
    from pypdf import PdfReader
    PYPDF_OK = True
except ImportError:
    PYPDF_OK = False
    logger.warning("pypdf not available — PDF text extraction disabled (install: pip install pypdf)")

try:
    from docx import Document as DocxDocument
    DOCX_OK = True
except ImportError:
    DOCX_OK = False
    logger.warning("python-docx not available — Word text extraction disabled (install: pip install python-docx)")

SUPPORTED_DOC_EXTENSIONS = {".pdf", ".docx", ".doc"}


def _pdf_str(val) -> Optional[str]:
    """Convert a pypdf metadata value (may be IndirectObject) to a plain str or None."""
    if val is None:
        return None
    try:
        s = str(val).strip()
        return s if s else None
    except Exception:
        return None
MAX_TEXT_CHARS = 50_000   # characters to extract per document for similarity comparison
MAX_DOC_SIZE_MB = 100


@dataclass
class DocumentRecord:
    """All extracted info about one document file."""
    path:         str
    filename:     str
    file_hash:    str          # SHA-256 — exact duplicate detection
    size_bytes:   int
    doc_type:     str          # "pdf" | "docx" | "doc"
    page_count:   int = 0
    word_count:   int = 0
    text_content: str = ""    # Extracted text (truncated to MAX_TEXT_CHARS)
    title:        Optional[str] = None
    author:       Optional[str] = None
    created_at:   Optional[datetime] = None
    modified_at:  Optional[datetime] = None
    error:        Optional[str] = None


class DocumentScanner:
    """
    Recursively scan a directory and return DocumentRecord objects.

    Usage:
        scanner = DocumentScanner("/path/to/docs")
        records = scanner.scan()
    """

    def __init__(self, root: str, recursive: bool = True):
        self.root      = Path(root)
        self.recursive = recursive

    def scan(self, progress_cb: Optional[Callable] = None) -> list[DocumentRecord]:
        paths   = self._collect_paths()
        total   = len(paths)
        records: list[DocumentRecord] = []
        completed = 0

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self._process_file, p): p for p in paths}
            for future in as_completed(futures):
                rec = future.result()
                if rec is not None:
                    records.append(rec)
                completed += 1
                if progress_cb:
                    progress_cb(
                        f"Scanning documents… ({completed}/{total})",
                        5 + int(80 * completed / max(total, 1)),
                        completed,
                        total,
                    )

        ok  = sum(1 for r in records if not r.error)
        err = sum(1 for r in records if r.error)
        logger.info(f"Document scan complete: {ok} OK, {err} errors")
        return records

    # ── Internal ──────────────────────────────────────────────────────────────
    def _collect_paths(self) -> list[Path]:
        pattern = "**/*" if self.recursive else "*"
        max_bytes = MAX_DOC_SIZE_MB * 1024 * 1024
        return [
            p for p in self.root.glob(pattern)
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_DOC_EXTENSIONS
            and p.stat().st_size <= max_bytes
        ]

    def _process_file(self, path: Path) -> Optional[DocumentRecord]:
        try:
            suffix     = path.suffix.lower()
            stat       = path.stat()
            file_hash  = self._file_hash(path)

            if suffix == ".pdf":
                return self._process_pdf(path, stat, file_hash)
            elif suffix in {".docx", ".doc"}:
                return self._process_docx(path, stat, file_hash)
        except Exception as e:
            logger.warning(f"Error processing document {path}: {e}")
            return DocumentRecord(
                path=str(path),
                filename=path.name,
                file_hash="",
                size_bytes=0,
                doc_type=path.suffix.lower().lstrip("."),
                error=str(e),
            )
        return None

    def _process_pdf(self, path: Path, stat, file_hash: str) -> DocumentRecord:
        rec = DocumentRecord(
            path=str(path),
            filename=path.name,
            file_hash=file_hash,
            size_bytes=stat.st_size,
            doc_type="pdf",
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
        )

        if not PYPDF_OK:
            return rec

        try:
            reader = PdfReader(str(path))
            rec.page_count = len(reader.pages)

            meta = reader.metadata
            if meta:
                rec.title  = _pdf_str(meta.get("/Title")  or meta.get("title"))
                rec.author = _pdf_str(meta.get("/Author") or meta.get("author"))
                # Try to parse creation date from PDF metadata
                pdf_date = meta.get("/CreationDate") or meta.get("creation_date")
                if pdf_date and isinstance(pdf_date, str) and pdf_date.startswith("D:"):
                    try:
                        rec.created_at = datetime.strptime(pdf_date[2:16], "%Y%m%d%H%M%S")
                    except ValueError:
                        pass

            parts: list[str] = []
            total_chars = 0
            for page in reader.pages:
                if total_chars >= MAX_TEXT_CHARS:
                    break
                try:
                    text = page.extract_text() or ""
                    remaining = MAX_TEXT_CHARS - total_chars
                    parts.append(text[:remaining])
                    total_chars += len(text)
                except Exception:
                    continue

            rec.text_content = " ".join(parts)
            rec.word_count   = len(rec.text_content.split())
        except Exception as e:
            logger.debug(f"PDF extraction error for {path}: {e}")

        return rec

    def _process_docx(self, path: Path, stat, file_hash: str) -> DocumentRecord:
        suffix = path.suffix.lower()
        rec = DocumentRecord(
            path=str(path),
            filename=path.name,
            file_hash=file_hash,
            size_bytes=stat.st_size,
            doc_type=suffix.lstrip("."),
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
        )

        # Old binary .doc format is not supported by python-docx
        if not DOCX_OK or suffix == ".doc":
            return rec

        try:
            doc   = DocxDocument(str(path))
            props = doc.core_properties

            rec.title  = props.title  or None
            rec.author = props.author or None
            if props.created:
                rec.created_at = props.created
            if props.modified:
                rec.modified_at = props.modified

            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text  = " ".join(paragraphs)
            rec.text_content = full_text[:MAX_TEXT_CHARS]
            rec.word_count   = len(rec.text_content.split())
        except Exception as e:
            logger.debug(f"DOCX extraction error for {path}: {e}")

        return rec

    @staticmethod
    def _file_hash(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(chunk))
        return h.hexdigest()
