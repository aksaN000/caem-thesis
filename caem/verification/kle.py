"""Kernel Language Entropy (KLE) signal.

Phase 2.2 (Branch C 2026-04-25): replaces the current ``s_avg`` and
``h_norm`` signals with a single principled uncertainty estimate.

KLE generalizes Farquhar et al.'s (2024 Nature) hard-cluster semantic
entropy with a soft NLI-similarity kernel and computes the von Neumann
entropy of the (normalized) kernel matrix. The result captures graded
relatedness across sampled chains rather than the discrete cluster
assignment of the original semantic-entropy formulation, and was shown
in the published evaluation to dominate Farquhar SE on factual-QA AUROC
across LLaMA-2 / Mistral / Falcon by 1–4 absolute points.

Empirical justification for replacing s_avg + h_norm with KLE
(Phase 1 Cycle-0 audit, n=3500 across 7 benchmarks):
  - s_avg Cohen's d sign-flips across benchmarks (-0.42 on TruthfulQA,
    +0.31 on TriviaQA, -0.10 on FEVER) → unreliable as a fixed-weight
    composite signal.
  - h_norm Cohen's d also sign-flips (-0.33 to +0.51 across the panel)
    and contributes near zero in the post-2026-04-24 weighted sum.
  - KLE strictly subsumes both: its kernel is the same NLI-similarity
    matrix that h_norm clusters and s_avg averages, but the von Neumann
    entropy formulation provides a single principled scalar with stable
    sign across benchmark formats.

Reference:
  Nikitin, Kossen, Gal, Janson. "Kernel Language Entropy: Fine-grained
  Uncertainty Quantification for LLMs from Semantic Similarities."
  NeurIPS 2024. arXiv:2405.20003.

Implementation:
  Given M sampled chains and an NLI bundle that computes pairwise
  bidirectional entailment probabilities, we build a similarity kernel
  K[i,j] = (p(i ⊨ j) + p(j ⊨ i)) / 2 with K[i,i] = 1, normalize to a
  density matrix ρ = K / tr(K), and compute von Neumann entropy
  H(ρ) = -Σ λ_k log λ_k where λ_k are the eigenvalues of ρ. Normalized
  by log(M) so output is in [0, 1].

  Drop-in: occupies the slot the existing s_avg + h_norm pair occupies in
  the verifier's batch_signals output. Composite consumers receive the
  new ``kle`` field; CalProbComposite picks it up via the same
  per-signal isotonic-regression mechanism that handles every other
  signal — no special casing.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def kle_from_kernel(kernel: np.ndarray) -> float:
    """Compute KLE from a precomputed similarity kernel.

    Parameters
    ----------
    kernel : np.ndarray, shape (M, M)
        Symmetric, [0, 1]-bounded similarity kernel. Diagonal must be ~1.

    Returns
    -------
    float
        KLE in [0, 1]. Higher = more diverse / uncertain. Lower = sampled
        chains are semantically tightly clustered → confident.
    """
    K = np.asarray(kernel, dtype=np.float64)
    M = K.shape[0]
    if M < 2:
        return 0.0  # single chain → degenerate, treat as zero entropy
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"kernel must be square, got shape {K.shape}")

    # Symmetrize defensively (NLI may produce slightly non-symmetric scores
    # due to direction asymmetry in the underlying judge).
    K = 0.5 * (K + K.T)
    # Normalize to a density matrix: ρ = K / tr(K)
    trace = float(np.trace(K))
    if trace <= 1e-9:
        return 0.0
    rho = K / trace
    # Eigenvalues of a symmetric matrix are real; clip negatives from
    # numerical noise.
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(eigvals, 1e-12, None)
    eigvals /= eigvals.sum()  # renormalize
    # von Neumann entropy = -Σ λ_k log λ_k
    H = float(-np.sum(eigvals * np.log(eigvals)))
    # Normalize by log(M), the maximum entropy of an M-dimensional density.
    H_norm = H / math.log(M) if M > 1 else 0.0
    return float(np.clip(H_norm, 0.0, 1.0))


def kle_from_chains(
    chains: Sequence[str],
    nli_judge: Any,
) -> float:
    """Compute KLE directly from sampled chains using an NLI judge.

    Parameters
    ----------
    chains : sequence of str
        Sampled chains for the same query (typically M=3 to M=10).
    nli_judge : object
        Must expose ``batch_entail_prob(pairs: List[Tuple[str, str]]) -> List[float]``
        — the same interface MiniCheck / generic-NLI bundles already provide
        elsewhere in the verifier.

    Returns
    -------
    float
        KLE in [0, 1].
    """
    M = len(chains)
    if M < 2:
        return 0.0
    # Build full bidirectional pair list (M^2 - M ordered pairs, excluding
    # self pairs which we set to 1.0 directly).
    pairs: List[Tuple[str, str]] = []
    pair_idx: List[Tuple[int, int]] = []
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            pairs.append((chains[i], chains[j]))
            pair_idx.append((i, j))
    try:
        scores = list(nli_judge.batch_entail_prob(pairs))
    except Exception as exc:
        logger.warning("kle_from_chains: NLI batch failed (%s) — returning 0.5", exc)
        return 0.5
    K = np.eye(M, dtype=np.float64)
    for (i, j), p in zip(pair_idx, scores):
        K[i, j] = float(np.clip(p, 0.0, 1.0))
    return kle_from_kernel(K)
