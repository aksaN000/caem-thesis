"""
caem/memory/encoder.py
======================
QueryEncoder: wraps Sentence-BERT (all-mpnet-base-v2) to produce
768-dim embeddings for episodic memory storage and retrieval.

Key facts from the thesis spec:
  - Model:  sentence-transformers/all-mpnet-base-v2
  - Output: np.ndarray shape (768,), dtype float32
  - CRITICAL: 768-dim, NOT 384 -- this matches the mpnet architecture.
  - Embeddings are L2-normalised to unit length so that inner product (IP)
    equals cosine similarity. This is required for FAISS IndexFlatIP correctness.
"""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

# Expected output dimension from all-mpnet-base-v2.
EXPECTED_DIM = 768


class QueryEncoder:
    """Encode natural-language queries into 768-dim Sentence-BERT embeddings.

    The encoder is loaded lazily on first use to avoid GPU memory allocation
    at import time. This is important when running experiments that instantiate
    a CAEMConfig without necessarily needing the encoder (e.g., unit tests
    with mock embeddings).

    Usage
    -----
    >>> encoder = QueryEncoder()
    >>> emb = encoder.encode("What is the capital of France?")
    >>> emb.shape
    (768,)
    >>> np.linalg.norm(emb)   # Should be ≈ 1.0 (L2-normalised)
    1.0
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str
            HuggingFace / sentence-transformers model identifier.
            Must be all-mpnet-base-v2 (768-dim) unless you also change
            CAEMConfig.embedding_dim and rebuild the FAISS index.
        device : str or None
            'cuda', 'cpu', or None (auto-detect). Auto-detection checks for
            CUDA availability and falls back to CPU.
        normalize : bool
            If True (default), L2-normalise every embedding so that FAISS
            inner product = cosine similarity.
        """
        self.model_name = model_name
        self.normalize = normalize
        self._model = None   # Loaded lazily.

        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

    def _load_model(self) -> None:
        """Load the sentence-transformers model (called on first encode call)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from e

        logger.info("Loading Sentence-BERT model: %s on %s", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)

        # Validate dimension immediately after loading.
        probe = self._model.encode("probe", normalize_embeddings=False)
        actual_dim = probe.shape[0]
        if actual_dim != EXPECTED_DIM:
            raise RuntimeError(
                f"Model '{self.model_name}' produces {actual_dim}-dim embeddings, "
                f"expected {EXPECTED_DIM}. "
                "If you changed the model, update CAEMConfig.embedding_dim "
                "and rebuild the FAISS index."
            )
        logger.info("Encoder loaded. Output dim: %d", actual_dim)

    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """Encode one or more texts into 768-dim float32 embeddings.

        Parameters
        ----------
        text : str or list of str
            A single query string or a batch.

        Returns
        -------
        np.ndarray
            - Single string -> shape (768,), dtype float32.
            - List of N strings -> shape (N, 768), dtype float32.
            Embeddings are L2-normalised (unit length) if self.normalize=True.
        """
        if self._model is None:
            self._load_model()

        single = isinstance(text, str)
        texts = [text] if single else text

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self.normalize,   # sentence-transformers handles L2 norm
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # Double-check normalisation (guard against library version quirks).
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Only re-normalise if any norm deviates meaningfully from 1.0.
            if not np.allclose(norms, 1.0, atol=1e-5):
                logger.debug("Re-normalising embeddings (norm deviation detected).")
                embeddings = embeddings / np.clip(norms, 1e-10, None)

        return embeddings[0] if single else embeddings

    def encode_for_storage(self, text: str) -> np.ndarray:
        """Convenience wrapper: encode a single query for storage in FAISS.

        Always returns shape (768,), dtype float32, L2-normalised.
        This is the canonical call site for the EpisodicMemoryStore.add() path.
        """
        emb = self.encode(text)
        if emb.shape[0] != 768:
            raise ValueError(
                f"Expected (768,) embedding, got {emb.shape}. "
                "Check that sentence-transformers model is 768-dim (mpnet)."
            )
        return emb

    @property
    def dim(self) -> int:
        """Return the embedding dimension (768)."""
        return EXPECTED_DIM
