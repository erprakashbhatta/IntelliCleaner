"""
core/embedder.py — CLIP Visual Embedder
Generates 512-dim visual embeddings for each image using OpenAI CLIP.
Falls back to a lightweight ResNet if CLIP is unavailable.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BATCH_SIZE
from scanner import ImageRecord

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """
    Wraps CLIP (or fallback) to generate image embeddings.

    Usage:
        embedder = CLIPEmbedder()
        records  = embedder.embed(records)   # fills record.embedding in-place
    """

    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        self.model      = None
        self.preprocess = None
        self.device     = device or self._best_device()
        self.dim        = 512
        self._load_model(model_name)

    # ── Public API ────────────────────────────────────────────────────────────
    def embed(self, records: list[ImageRecord]) -> list[ImageRecord]:
        """Fill record.embedding for all records (cached or batch-computed)."""
        from scan_cache import get_cache
        cache = get_cache()

        valid     = [r for r in records if r.error is None]
        to_embed  = []

        # Load embeddings from cache where available
        for r in valid:
            if r.embedding is not None:   # already loaded by scanner cache hit
                continue
            try:
                stat   = Path(r.path).stat()
                cached = cache.get(r.path, stat.st_size, stat.st_mtime)
                if cached and cached.get("embedding"):
                    r.embedding = cached["embedding"]
                else:
                    to_embed.append(r)
            except OSError:
                to_embed.append(r)

        cache_hits = len(valid) - len(to_embed)
        logger.info(f"Embeddings: {cache_hits} from cache, {len(to_embed)} to compute")

        for i in tqdm(range(0, len(to_embed), BATCH_SIZE), desc="Embedding batches"):
            batch = to_embed[i : i + BATCH_SIZE]
            embs  = self._embed_batch(batch)
            for rec, emb in zip(batch, embs):
                rec.embedding = emb.tolist() if emb is not None else None
                # Save to cache
                try:
                    stat = Path(rec.path).stat()
                    cache.merge(rec.path, stat.st_size, stat.st_mtime,
                                {"embedding": rec.embedding})
                except OSError:
                    pass

        return records

    def embed_single(self, image: Image.Image) -> Optional[list[float]]:
        """Embed a single PIL image. Returns list[float] or None."""
        try:
            result = self._embed_batch_images([image])
            return result[0].tolist() if result and result[0] is not None else None
        except Exception as e:
            logger.warning(f"Single embed failed: {e}")
            return None

    # ── Internal ──────────────────────────────────────────────────────────────
    def _load_model(self, model_name: str):
        try:
            import clip
            import torch
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
            self._backend = "clip"
            logger.info(f"CLIP model loaded: {model_name} on {self.device}")
        except ImportError:
            logger.warning("openai-clip not installed, falling back to torchvision ResNet")
            self._load_resnet_fallback()

    def _load_resnet_fallback(self):
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T

            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            resnet.fc = torch.nn.Identity()   # strip classifier head → 2048-dim
            resnet.eval()
            self.model = resnet.to(self.device)
            self.dim   = 2048

            self.preprocess = T.Compose([
                T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._backend = "resnet"
            logger.info(f"ResNet50 fallback loaded on {self.device}")
        except Exception as e:
            logger.error(f"All embedding backends failed: {e}")
            self.model     = None
            self._backend  = "none"

    def _embed_batch(self, records: list[ImageRecord]) -> list[Optional[np.ndarray]]:
        images: list[Optional[Image.Image]] = []
        for r in records:
            try:
                images.append(Image.open(r.path).convert("RGB"))
            except Exception:
                images.append(None)
        return self._embed_batch_images(images)

    def _embed_batch_images(
        self, images: list[Optional[Image.Image]]
    ) -> list[Optional[np.ndarray]]:
        if self.model is None or self._backend == "none":
            return [None] * len(images)

        import torch

        valid_idx, valid_imgs = [], []
        for i, img in enumerate(images):
            if img is not None:
                valid_idx.append(i)
                valid_imgs.append(img)

        if not valid_imgs:
            return [None] * len(images)

        try:
            with torch.no_grad():
                if self._backend == "clip":
                    import clip
                    batch = torch.stack([self.preprocess(img) for img in valid_imgs]).to(self.device)
                    feats = self.model.encode_image(batch)
                else:
                    batch = torch.stack([self.preprocess(img) for img in valid_imgs]).to(self.device)
                    feats = self.model(batch)

            feats = feats.cpu().float().numpy()
            # L2-normalise
            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
            feats = feats / norms
        except Exception as e:
            logger.warning(f"Batch embedding error: {e}")
            return [None] * len(images)

        result: list[Optional[np.ndarray]] = [None] * len(images)
        for idx, emb in zip(valid_idx, feats):
            result[idx] = emb
        return result

    @staticmethod
    def _best_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
