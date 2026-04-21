"""
caem/training/loop_filter.py
============================
Two-signal repetitive-loop detector for verified reasoning chains.

Motivation (Branch C Goal 4)
----------------------------
Phase-1a Cycle-0 measurement showed ~30% of STOREd samples were severe
repetition loops (distinct-4 <= 0.13 on 82/241 STOREd samples), with rates
up to 52% on FEVER and 56% on StrategyQA. Loops passed the pre-fix
verifier because high-confidence repetition produces high p_entail and
high u_internal while still being unusable as supervised training data.

Left unfiltered, these loops poison the SIL training pool:
``_collect_episodes`` regresses the model onto ``"yes yes yes ..."``-style
reasoning chains, which can collapse open-ended generation quality.

Fix (model-agnostic hygiene, no theorem impact)
-----------------------------------------------
Two independent signals:

1. ``distinct-4`` -- the ratio of unique 4-grams to total 4-grams over the
   whitespace-tokenised chain. Below ~0.25 is the classical loop signature
   (Wei et al. "Finetuned Language Models are Zero-Shot Learners" 2021
   appendix, also standard in NLG-degeneration literature).

2. ``compression ratio`` -- ``len(zlib(text)) / len(text)``. Char-level
   loops compress far harder than prose. Below ~0.35 catches loops that
   evade the token-level test (e.g., character-level repetition, long
   whitespace runs).

A chain is flagged as a loop when EITHER signal crosses its threshold
AND the chain is long enough to judge (>= ``loop_min_tokens``). Short
answers (<20 tokens) legitimately have low distinct-n and are skipped.

All three thresholds are [DES] (not [LIT]) -- see ``CAEMConfig`` defaults
and hyperparameter-reference.md.
"""

from __future__ import annotations

import zlib

from caem.config import CAEMConfig


def _distinct_ngram_ratio(tokens: list, n: int = 4) -> float:
    """Return the fraction of unique n-grams in ``tokens``.

    Defined as ``len(set(ngrams)) / len(ngrams)``. Returns 1.0 for
    sequences too short to form any n-grams (cannot be a loop by
    construction).
    """
    if len(tokens) < n:
        return 1.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 1.0
    return len(set(ngrams)) / len(ngrams)


def _compression_ratio(text: str) -> float:
    """Return ``len(zlib(text)) / len(text)``; 1.0 for empty input.

    Low values (~<0.35) indicate char-level repetition that would shrink
    under general-purpose compression. Prose and coherent CoT compress to
    ~0.50-0.70 depending on the tokenizer.
    """
    if not text:
        return 1.0
    raw = text.encode("utf-8", errors="replace")
    if not raw:
        return 1.0
    compressed = zlib.compress(raw)
    return len(compressed) / len(raw)


def is_repetitive_loop(text: str, config: CAEMConfig) -> bool:
    """Return True iff ``text`` crosses either loop-detection threshold.

    Short texts below ``config.loop_min_tokens`` are always judged clean
    -- a 3-word answer legitimately has distinct-4 = 0 / 0 with no loop.

    Both signals are computed independently; the chain is a loop when
    EITHER fires. This mirrors the OR-of-safety-checks convention used
    elsewhere in the verifier (e.g., the contradiction veto in Stage 5).
    """
    if not text:
        return False
    s = text.strip()
    if not s:
        return False

    tokens = s.split()
    if len(tokens) < config.loop_min_tokens:
        return False

    distinct4 = _distinct_ngram_ratio(tokens, n=4)
    if distinct4 < config.loop_distinct4_threshold:
        return True

    compression = _compression_ratio(s)
    if compression < config.loop_compression_threshold:
        return True

    return False


__all__ = ["is_repetitive_loop"]
