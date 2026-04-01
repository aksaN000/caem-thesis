"""
caem/confidence/post_generation.py
====================================
PostGenerationConfidenceEstimator — Stage 4a of the CAEM pipeline.

Runs in Tier 2 ONLY, AFTER a full answer has been generated.

Purpose — EFFICIENCY GATE, not a quality gate
----------------------------------------------
Stage 4a decides whether to commit to the expensive verification pipeline
(Stage 5: NLI + SC + SE, ~2–4 s). It catches obvious low-confidence
outputs early, before spending compute on them.

    if û ≥ 0.60 (CAEMConfig.u_hat_accept_threshold):
        → accept → pass to Stage 5 (MultiLayerVerifier)
    else:
        → escalate to Tier 3 (full RAG generation)

Stage 4a does NOT determine answer quality — Stage 5 does. See writing-
suggestions.md C4-05.

Four signals
------------
All four are literature-fixed in terms of their methodology; only the
combination weights are calibrated.

Signal 1 — u_token  [LIT: geometric mean of token log-probs]
    Recomputed on the actual generated answer (not a short greedy prefix
    as in Stage 3). Captures per-token generation confidence.

Signal 2 — u_dropout  [LIT: Gal & Ghahramani 2016, K=5 MC Dropout passes]
    Variance of output probabilities across K stochastic forward passes
    with dropout enabled at inference.
    u_dropout = 1 / (1 + Var[answer_probs across K passes])
    High variance = model is uncertain about its own answer.

Signal 3 — u_consistency  [LIT: Wang et al. 2022, M=3 chains]
    Average pairwise cosine similarity of M independent chain-of-thought
    generations. High similarity = model produces consistent reasoning.
    Uses Sentence-BERT (same encoder as the memory store).

Signal 4 — u_entropy  [LIT: Farquhar et al. 2024, K=10 samples at T=1.0]
    Semantic entropy over K stochastic samples. Samples are clustered by
    bidirectional NLI entailment (same meaning = same cluster). Entropy
    over the cluster distribution measures meaning-level uncertainty, not
    just surface-level token variance.
    u_entropy = 1 − H_semantic / log2(K)

Combination weights
-------------------
Initial (implementation starting point):  0.25 / 0.25 / 0.25 / 0.25
Projected post-calibration:               0.20 / 0.20 / 0.20 / 0.40
    (SE upweighted; Farquhar et al. 2024 AUROC ≈ 0.79)
Actual calibrated weights: fitted on 500-sample calibration set after
Cycle 1 and reported in Chapter 5. See: hyperparameter-reference.md.

DO NOT initialise at 0.20/0.20/0.20/0.40 — start equal at 0.25 each.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from caem.config import CAEMConfig
from caem.memory.entry import PostGenerationConfidence

logger = logging.getLogger(__name__)


class PostGenerationConfidenceEstimator:
    """Compute û (post-generation confidence) for a Tier 2 answer.

    Parameters
    ----------
    model : transformers.T5ForConditionalGeneration
        Flan-T5-Large in eval mode. Temporarily switched to train() for
        MC Dropout passes, then restored to eval().
    tokenizer : transformers.AutoTokenizer
        Matching tokenizer.
    sbert_encoder : QueryEncoder
        Sentence-BERT encoder for u_consistency cosine similarity.
        Must already be initialised (shared with EpisodicMemoryStore).
    nli_model : optional
        RoBERTa-Large-MNLI for semantic entropy clustering.
        If None, u_entropy falls back to 0.5 (neutral — no signal).
        Pass the actual model once available for full SE computation.
    nli_tokenizer : optional
        Tokenizer for nli_model.
    config : CAEMConfig
        Weights and thresholds. u_hat weights start at 0.25 each.
    device : str or None
        Auto-detected from model if None.

    Usage
    -----
    >>> est = PostGenerationConfidenceEstimator(model, tokenizer, encoder)
    >>> pgc = est.estimate(query="Who wrote Hamlet?",
    ...                    generated_answer="William Shakespeare",
    ...                    input_ids=input_ids)
    >>> pgc.u_hat          # e.g. 0.72
    >>> pgc.should_accept()  # True → send to verifier; False → escalate Tier 3
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

    def estimate(
        self,
        query: str,
        generated_answer: str,
        input_ids: Optional[torch.Tensor] = None,
    ) -> PostGenerationConfidence:
        """Compute û for a Tier 2 (query, generated_answer) pair.

        Parameters
        ----------
        query : str
            The original query text.
        generated_answer : str
            The answer already generated by Tier 2.
        input_ids : torch.Tensor or None
            Pre-tokenized input for the model. If None, query is tokenized
            internally. Passing pre-computed input_ids avoids double tokenization
            when the caller already has them from Tier 2 generation.

        Returns
        -------
        PostGenerationConfidence
            All four signal values + û. Call .should_accept() to gate.
        """
        if input_ids is None:
            input_ids = self._tokenize(query)["input_ids"]

        with torch.no_grad():
            u_token = self._compute_u_token(input_ids, generated_answer)

        u_dropout     = self._compute_u_dropout(input_ids)
        u_consistency = self._compute_u_consistency(query, input_ids)
        u_entropy     = self._compute_u_entropy(query, input_ids)

        cfg = self.config
        u_hat = (
            cfg.u_hat_weight_token    * u_token
            + cfg.u_hat_weight_dropout  * u_dropout
            + cfg.u_hat_weight_sc       * u_consistency
            + cfg.u_hat_weight_entropy  * u_entropy
        )
        u_hat = float(np.clip(u_hat, 0.0, 1.0))

        logger.debug(
            "û estimate | u_token=%.4f | u_dropout=%.4f | u_sc=%.4f | "
            "u_entropy=%.4f | û=%.4f | accept=%s",
            u_token, u_dropout, u_consistency, u_entropy, u_hat,
            u_hat >= cfg.u_hat_accept_threshold,
        )

        return PostGenerationConfidence(
            u_token=float(u_token),
            u_dropout=float(u_dropout),
            u_consistency=float(u_consistency),
            u_entropy=float(u_entropy),
            u_hat=u_hat,
        )

    # ------------------------------------------------------------------ #
    # Signal 1 — u_token                                                  #
    # ------------------------------------------------------------------ #

    def _compute_u_token(
        self,
        input_ids: torch.Tensor,
        generated_answer: str,
    ) -> float:
        """Geometric mean of per-token log-probs for the generated answer.

        Unlike Stage 3 (short greedy prefix for speed), here we score the
        actual full generated answer token-by-token. This gives a more
        faithful confidence estimate for the complete response.

        Returns float in [0, 1]. Returns 0.0 on error (pessimistic fail).
        """
        try:
            # Transformers target-tokenization API changed from
            # as_target_tokenizer() to text_target=...; support both.
            tok_kwargs = {
                "return_tensors": "pt",
                "truncation": True,
                "max_length": 256,
            }
            try:
                target_tokens = self.tokenizer(text_target=generated_answer, **tok_kwargs)
            except TypeError:
                if hasattr(self.tokenizer, "as_target_tokenizer"):
                    with self.tokenizer.as_target_tokenizer():
                        target_tokens = self.tokenizer(generated_answer, **tok_kwargs)
                else:
                    target_tokens = self.tokenizer(generated_answer, **tok_kwargs)

            if hasattr(target_tokens, "input_ids"):
                answer_ids = target_tokens.input_ids.to(self.device)
            else:
                answer_ids = target_tokens["input_ids"].to(self.device)

            # Forward pass: model scores each decoder token given the input
            outputs = self.model(
                input_ids=input_ids,
                labels=answer_ids,
                output_hidden_states=False,
            )
            # logits: (batch=1, seq_len, vocab_size)
            logits = outputs.logits[0]   # (seq_len, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1)

            # Collect log P(chosen token) for each position
            # answer_ids[0] has shape (seq_len,); ignore padding (id=0) and EOS
            chosen = answer_ids[0]
            token_log_probs = []
            for t, tok_id in enumerate(chosen):
                if tok_id.item() in (self.tokenizer.pad_token_id, self.tokenizer.eos_token_id):
                    continue
                if t >= logits.shape[0]:
                    break
                token_log_probs.append(log_probs[t, tok_id.item()].item())

            if not token_log_probs:
                return 0.5

            u_token = math.exp(sum(token_log_probs) / len(token_log_probs))
            return float(np.clip(u_token, 0.0, 1.0))

        except Exception as exc:
            logger.warning("u_token (post-gen): failed with %s — returning 0.0.", exc)
            return 0.0

    # ------------------------------------------------------------------ #
    # Signal 2 — u_dropout (MC Dropout, K=5)                              #
    # ------------------------------------------------------------------ #

    def _compute_u_dropout(self, input_ids: torch.Tensor) -> float:
        """MC Dropout uncertainty: 1 / (1 + Var[answer_probs]) over K=5 passes.

        Method:
          1. Switch model to train() mode → dropout layers activate.
          2. Run K independent forward passes (stochastic due to dropout).
          3. Collect the mean token probability of each generation.
          4. Variance across K values → uncertainty.
          5. Restore model to eval() mode.

        K=5 is fixed from Gal & Ghahramani (2016) [LIT].
        Returns float in [0, 1]. Returns 0.5 (neutral) on error.
        """
        K = self.config.mc_dropout_k
        answer_probs = []

        try:
            self.model.train()   # Enable dropout
            with torch.no_grad():
                for _ in range(K):
                    out = self.model.generate(
                        input_ids,
                        max_new_tokens=self.config.cot_max_new_tokens,
                        do_sample=False,       # Greedy — dropout is the only stochasticity
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                    if not out.scores:
                        answer_probs.append(0.5)
                        continue
                    # Mean token probability for this pass
                    pass_log_probs = []
                    for t, step_scores in enumerate(out.scores):
                        log_sm = F.log_softmax(step_scores[0], dim=-1)
                        tok_id = out.sequences[0, t + 1].item()
                        if tok_id == self.tokenizer.eos_token_id:
                            break
                        pass_log_probs.append(log_sm[tok_id].item())
                    if pass_log_probs:
                        answer_probs.append(math.exp(sum(pass_log_probs) / len(pass_log_probs)))
                    else:
                        answer_probs.append(0.5)
        except Exception as exc:
            logger.warning("u_dropout: failed with %s — returning 0.5.", exc)
            return 0.5
        finally:
            self.model.eval()   # ALWAYS restore eval mode

        if len(answer_probs) < 2:
            return 0.5

        variance = float(torch.var(torch.tensor(answer_probs)).item())
        u_dropout = 1.0 / (1.0 + variance)
        return float(np.clip(u_dropout, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Signal 3 — u_consistency (M=3 chain-of-thought chains)              #
    # ------------------------------------------------------------------ #

    def _compute_u_consistency(
        self,
        query: str,
        input_ids: torch.Tensor,
    ) -> float:
        """Average pairwise cosine similarity of M=3 sampled generations.

        M=3 is fixed from Wang et al. (2022) [LIT].
        Sampling temperature=0.7 gives diverse but coherent chains.
        Cosine similarity uses Sentence-BERT (same as memory store).

        Returns float in [0, 1]. Returns 0.5 (neutral) on error.
        """
        M = self.config.sc_chains_m
        try:
            self.model.eval()
            with torch.no_grad():
                chains = []
                for _ in range(M):
                    out = self.model.generate(
                        input_ids,
                        max_new_tokens=self.config.cot_max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                    )
                    decoded = self.tokenizer.decode(
                        out[0], skip_special_tokens=True
                    )
                    chains.append(decoded)

            if len(chains) < 2:
                return 0.5

            # Encode all chains with Sentence-BERT
            embeddings = self.sbert_encoder.encode(chains)   # (M, 384)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            # Average pairwise cosine similarity
            # (Embeddings are L2-normalised → cosine sim = dot product)
            sims = []
            for i, j in itertools.combinations(range(len(embeddings)), 2):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                sims.append(sim)

            u_consistency = float(np.mean(sims)) if sims else 0.5
            return float(np.clip(u_consistency, 0.0, 1.0))

        except Exception as exc:
            logger.warning("u_consistency: failed with %s — returning 0.5.", exc)
            return 0.5

    # ------------------------------------------------------------------ #
    # Signal 4 — u_entropy (Semantic Entropy, Farquhar et al. 2024)       #
    # ------------------------------------------------------------------ #

    def _compute_u_entropy(
        self,
        query: str,
        input_ids: torch.Tensor,
    ) -> float:
        """Semantic entropy over K=10 samples at T=1.0.

        Method (Farquhar et al. 2024) [LIT]:
          1. Sample K=10 answers at T=1.0 (maximum diversity).
          2. Cluster by bidirectional NLI entailment:
             Two answers are in the same semantic cluster iff
             NLI(a→b) = ENTAILMENT  AND  NLI(b→a) = ENTAILMENT.
          3. Compute entropy over cluster size distribution:
             H = -Σ p_c · log2(p_c),   p_c = |cluster c| / K
          4. Normalise: Ĥ = H / log2(K)
          5. u_entropy = 1 − Ĥ

        If nli_model is None, falls back to surface-level distinct-answer
        entropy (less accurate but avoids a hard dependency on RoBERTa at
        import time). Fallback is documented as [DES-fallback] in log.

        Returns float in [0, 1]. Returns 0.5 (neutral) on error.
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
                        max_new_tokens=self.config.cot_max_new_tokens,
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
                H_sem = self._semantic_entropy_nli(samples, query)
            else:
                # Fallback: surface-level distinct-answer entropy
                # Less accurate — NLI-based clustering is the correct method.
                # This branch is used during development before RoBERTa is loaded.
                logger.debug("u_entropy: NLI model not available — using surface fallback.")
                H_sem = self._semantic_entropy_surface(samples)

            h_norm = H_sem / math.log2(max(K, 2))
            u_entropy = 1.0 - h_norm
            return float(np.clip(u_entropy, 0.0, 1.0))

        except Exception as exc:
            logger.warning("u_entropy: failed with %s — returning 0.5.", exc)
            return 0.5

    def _semantic_entropy_nli(self, samples: List[str], query: str) -> float:
        """Bidirectional NLI clustering → Shannon entropy (bits).

        Two samples are semantically equivalent iff:
          NLI(sample_i → sample_j) = ENTAILMENT
          AND
          NLI(sample_j → sample_i) = ENTAILMENT

        Union-Find is used to group samples into semantic clusters.
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
            """Return argmax over [CONTRADICTION=0, NEUTRAL=1, ENTAILMENT=2]."""
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
                if nli_label(samples[i], samples[j]) == ENTAILMENT and \
                   nli_label(samples[j], samples[i]) == ENTAILMENT:
                    union(i, j)

        # Count cluster sizes
        cluster_counts: dict = {}
        for i in range(N):
            root = find(i)
            cluster_counts[root] = cluster_counts.get(root, 0) + 1

        # Shannon entropy
        H = 0.0
        for count in cluster_counts.values():
            p = count / N
            if p > 0:
                H -= p * math.log2(p)
        return H

    def _semantic_entropy_surface(self, samples: List[str]) -> float:
        """Surface-level entropy fallback (used when NLI model is unavailable).

        Groups by exact string match after lowercasing and stripping punctuation.
        Significantly less accurate than NLI clustering for paraphrases — use
        only during development. Logged as [DES-fallback] in implementation log.
        """
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
