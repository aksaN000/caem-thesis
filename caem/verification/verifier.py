"""
caem/verification/verifier.py
==============================
MultiLayerVerifier — Stage 5 of the CAEM pipeline.

Runs after a Tier 2 answer passes Stage 4a (û ≥ 0.60).
Determines whether the answer is trustworthy enough to store in episodic
memory and at what confidence level (û_stored).

Three signals
-------------
Signal 1 — p_entail  [DES]
    NLI entailment probability: P(ENTAILMENT | query, answer) from the NLI
    model's softmax output (NOT just argmax). Captures factual consistency
    between the question and the generated answer.

    Implementation: for each of M independently generated chains, compute
    P(ENTAILMENT) of (chain → original_answer), then average. This checks
    that the model's own reasoning chains consistently support the answer,
    rather than trusting the NLI model's relationship between a raw question
    and an opaque answer string.

Signal 2 — s_avg  [LIT: Wang et al. 2022]
    Average pairwise cosine similarity of M=3 independent chain-of-thought
    generations (Sentence-BERT, 384-dim, same encoder as memory store).
    High similarity = the model's reasoning is stable; low = uncertain or
    multi-modal answer space.

Signal 3 — h_norm  [LIT: Farquhar et al. 2024]
    Normalised semantic entropy from K=10 samples at T=1.0, using
    bidirectional NLI clustering (same method as Stage 4a u_entropy).
    Measures meaning-level, not surface-level, diversity.

Combination
-----------
    û_stored = 0.50·p_entail + 0.30·s_avg + 0.20·(1 − h_norm)

    Weights [DES]: NLI entailment is the strongest post-hoc signal (0.50),
    self-consistency is secondary (0.30), semantic entropy contributes the
    remainder (0.20). See: hyperparameter-reference.md u_stored_weight_*.

Distinction from Stage 4a
--------------------------
Stage 4a (PostGenerationConfidenceEstimator) is an EFFICIENCY gate:
it prevents sending low-confidence answers to expensive Stage 5 compute.
Stage 5 (MultiLayerVerifier) is a QUALITY gate: it determines what û_stored
value to write into the episodic memory entry. These two stages are
deliberately separate — Stage 4a can be calibrated for recall (avoid false
negatives) while Stage 5 is calibrated for precision (avoid storing junk).
See: writing-suggestions.md C4-05.

If û_stored ≥ retroverify_prune_threshold (default 0.50):
    → Write to episodic memory (Stage 7).
If û_stored < threshold:
    → Discard — not stored.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from caem.config import CAEMConfig
from caem.memory.entry import StoredConfidence

logger = logging.getLogger(__name__)


class MultiLayerVerifier:
    """Compute û_stored for a (query, answer) pair.

    Parameters
    ----------
    model : transformers.T5ForConditionalGeneration
        Flan-T5-Large in eval mode. Used for M-chain generation (s_avg)
        and K-sample generation (h_norm).
    tokenizer : transformers.AutoTokenizer
        Matching tokenizer for the main model.
    sbert_encoder : QueryEncoder
        Sentence-BERT encoder (shared with EpisodicMemoryStore).
    nli_model : optional
        RoBERTa-Large-MNLI for p_entail and h_norm NLI clustering.
        If None, p_entail falls back to 0.5 and h_norm uses surface entropy.
    nli_tokenizer : optional
        Tokenizer for nli_model.
    config : CAEMConfig
    device : str or None

    Usage
    -----
    >>> verifier = MultiLayerVerifier(model, tokenizer, encoder,
    ...                               nli_model, nli_tokenizer)
    >>> sc = verifier.verify("Who wrote Hamlet?", "William Shakespeare")
    >>> sc.u_stored   # e.g. 0.82
    >>> sc.u_stored >= 0.50  # True → store in memory
    """

    def __init__(
        self,
        model,
        tokenizer,
        sbert_encoder,
        nli_model=None,
        nli_tokenizer=None,
        config: Optional[CAEMConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sbert_encoder = sbert_encoder
        self.nli_model = nli_model
        self.nli_tokenizer = nli_tokenizer
        self.config = config or CAEMConfig()

        if device is None:
            device = str(next(model.parameters()).device)
        self.device = device

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def verify(
        self,
        query: str,
        answer: str,
        input_ids: Optional[torch.Tensor] = None,
    ) -> StoredConfidence:
        """Compute û_stored for a (query, answer) pair.

        Parameters
        ----------
        query : str
            The original query text.
        answer : str
            The generated answer that passed Stage 4a.
        input_ids : torch.Tensor or None
            Pre-tokenized query. If None, tokenized internally.

        Returns
        -------
        StoredConfidence
            p_entail, s_avg, h_norm, û_stored.
            Check sc.u_stored >= config.retroverify_prune_threshold to decide
            whether to store.
        """
        if input_ids is None:
            input_ids = self._tokenize(query)["input_ids"]

        p_entail = self._compute_p_entail(query, answer, input_ids)
        s_avg    = self._compute_s_avg(query, answer, input_ids)
        h_norm   = self._compute_h_norm(query, input_ids)

        cfg = self.config
        u_stored = (
            cfg.u_stored_weight_nli * p_entail
            + cfg.u_stored_weight_sc  * s_avg
            + cfg.u_stored_weight_se  * (1.0 - h_norm)
        )
        u_stored = float(np.clip(u_stored, 0.0, 1.0))

        logger.debug(
            "verify | p_entail=%.4f | s_avg=%.4f | h_norm=%.4f | û_stored=%.4f | store=%s",
            p_entail, s_avg, h_norm, u_stored,
            u_stored >= cfg.retroverify_prune_threshold,
        )

        return StoredConfidence(
            p_entail=float(p_entail),
            s_avg=float(s_avg),
            h_norm=float(h_norm),
            u_stored=u_stored,
        )

    def should_store(self, sc: StoredConfidence) -> bool:
        """Return True if û_stored meets the storage threshold."""
        return sc.u_stored >= self.config.retroverify_prune_threshold

    # ------------------------------------------------------------------ #
    # Signal 1 — p_entail                                                  #
    # ------------------------------------------------------------------ #

    def _compute_p_entail(
        self,
        query: str,
        answer: str,
        input_ids: torch.Tensor,
    ) -> float:
        """P(ENTAILMENT) averaged over M independent model generations.

        For each of M chains generated from the query, compute the NLI
        probability that the chain ENTAILS the given answer. High average
        p_entail means the model's own reasoning consistently supports
        the answer — a strong factual reliability signal.

        If NLI model is unavailable, falls back to 0.5 (neutral).
        Returns float in [0, 1].
        """
        if self.nli_model is None or self.nli_tokenizer is None:
            logger.debug("p_entail: NLI model unavailable — returning 0.5.")
            return 0.5

        M = self.config.sc_chains_m
        try:
            self.model.eval()
            with torch.no_grad():
                chains = []
                for _ in range(M):
                    out = self.model.generate(
                        input_ids,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.7,
                    )
                    decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
                    chains.append(decoded)

            entail_probs = []
            for chain in chains:
                p = self._nli_entail_prob(premise=chain, hypothesis=answer)
                entail_probs.append(p)

            p_entail = float(np.mean(entail_probs)) if entail_probs else 0.5
            return float(np.clip(p_entail, 0.0, 1.0))

        except Exception as exc:
            logger.warning("p_entail: failed with %s — returning 0.5.", exc)
            return 0.5

    def _nli_entail_prob(self, premise: str, hypothesis: str) -> float:
        """Softmax P(ENTAILMENT) from the NLI model for a (premise, hypothesis) pair.

        Unlike Stage 4a which uses argmax label for clustering, here we use
        the full probability to get a graded signal in [0, 1].
        Label order: [CONTRADICTION=0, NEUTRAL=1, ENTAILMENT=2].
        """
        enc = self.nli_tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.nli_model(**enc).logits      # (1, 3)
        probs = F.softmax(logits, dim=-1)              # (1, 3)
        return float(probs[0, 2].item())               # ENTAILMENT index = 2

    # ------------------------------------------------------------------ #
    # Signal 2 — s_avg (self-consistency)                                  #
    # ------------------------------------------------------------------ #

    def _compute_s_avg(
        self,
        query: str,
        answer: str,
        input_ids: torch.Tensor,
    ) -> float:
        """Average pairwise cosine similarity of M=3 independent generations.

        M=3 [LIT: Wang et al. 2022]. Similarity computed via Sentence-BERT
        (same 384-dim encoder as episodic memory store).
        Returns float in [0, 1]. Returns 0.5 on error (neutral).
        """
        M = self.config.sc_chains_m
        try:
            self.model.eval()
            with torch.no_grad():
                chains = []
                for _ in range(M):
                    out = self.model.generate(
                        input_ids,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.7,
                    )
                    decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
                    chains.append(decoded)

            if len(chains) < 2:
                return 0.5

            embeddings = self.sbert_encoder.encode(chains)   # (M, 384)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            sims = []
            for i, j in itertools.combinations(range(len(embeddings)), 2):
                sims.append(float(np.dot(embeddings[i], embeddings[j])))

            s_avg = float(np.mean(sims)) if sims else 0.5
            return float(np.clip(s_avg, 0.0, 1.0))

        except Exception as exc:
            logger.warning("s_avg: failed with %s — returning 0.5.", exc)
            return 0.5

    # ------------------------------------------------------------------ #
    # Signal 3 — h_norm (semantic entropy)                                 #
    # ------------------------------------------------------------------ #

    def _compute_h_norm(
        self,
        query: str,
        input_ids: torch.Tensor,
    ) -> float:
        """Normalised semantic entropy H / log2(K) over K=10 samples at T=1.0.

        Uses bidirectional NLI entailment clustering (Farquhar et al. 2024)
        when the NLI model is available. Falls back to surface string-match
        entropy when not. Returns float in [0, 1].
        """
        K = self.config.se_samples_k
        T = self.config.se_temperature

        try:
            self.model.eval()
            with torch.no_grad():
                samples = []
                for _ in range(K):
                    out = self.model.generate(
                        input_ids,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=T,
                    )
                    decoded = self.tokenizer.decode(
                        out[0], skip_special_tokens=True
                    ).strip()
                    samples.append(decoded)

            if not samples:
                return 0.5

            if self.nli_model is not None and self.nli_tokenizer is not None:
                H = self._semantic_entropy_nli(samples)
            else:
                logger.debug("h_norm: NLI model unavailable — using surface fallback.")
                H = self._semantic_entropy_surface(samples)

            h_norm = H / math.log2(max(K, 2))
            return float(np.clip(h_norm, 0.0, 1.0))

        except Exception as exc:
            logger.warning("h_norm: failed with %s — returning 0.5.", exc)
            return 0.5

    # ------------------------------------------------------------------ #
    # NLI clustering (shared by p_entail and h_norm)                       #
    # ------------------------------------------------------------------ #

    def _semantic_entropy_nli(self, samples: List[str]) -> float:
        """Bidirectional NLI clustering → Shannon entropy (bits).

        Two samples are in the same semantic cluster iff both directions
        return ENTAILMENT. Union-Find merges clusters incrementally.
        Identical to the method in Stage 4a, but called from the verifier
        context (query is not passed; clustering is over samples only).
        """
        N = len(samples)
        parent = list(range(N))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        def nli_label(premise: str, hypothesis: str) -> int:
            enc = self.nli_tokenizer(
                premise, hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                logits = self.nli_model(**enc).logits
            return int(torch.argmax(logits, dim=-1).item())

        ENTAILMENT = 2
        for i in range(N):
            for j in range(i + 1, N):
                if (nli_label(samples[i], samples[j]) == ENTAILMENT and
                        nli_label(samples[j], samples[i]) == ENTAILMENT):
                    union(i, j)

        cluster_counts: dict = {}
        for i in range(N):
            root = find(i)
            cluster_counts[root] = cluster_counts.get(root, 0) + 1

        H = 0.0
        for count in cluster_counts.values():
            p = count / N
            if p > 0:
                H -= p * math.log2(p)
        return H

    def _semantic_entropy_surface(self, samples: List[str]) -> float:
        """Surface-level entropy fallback (string match, no NLI model required)."""
        import re

        def normalise(s: str) -> str:
            return re.sub(r"[^\w\s]", "", s.lower()).strip()

        counts: dict = {}
        for s in samples:
            key = normalise(s)
            counts[key] = counts.get(key, 0) + 1

        N = len(samples)
        H = 0.0
        for count in counts.values():
            p = count / N
            if p > 0:
                H -= p * math.log2(p)
        return H

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _tokenize(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
