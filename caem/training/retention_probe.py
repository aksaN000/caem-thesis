"""
caem/training/retention_probe.py
==================================
v2 Fix 4 — multi-modal retention probe.

v1 used a single MMLU-validation 200-sample probe to gate the SIL
forgetting check. The probe was blind to two failure modes that
showed up empirically in cycles 0-4:

  * **Open-text generation degradation** — MMLU is a 4-choice MCQ,
    so the probe scores letter accuracy. A model that learns to
    pick correct letters but produces garbage on free-form answers
    (TriviaQA / NQ / HotpotQA) sails through the MMLU guard while
    silently catastrophic-forgetting the open-domain QA distribution
    that the v2 panel actually tests.
  * **Multi-hop reasoning degradation** — single-hop benchmarks (MMLU,
    TriviaQA) do not exercise the chain-of-reasoning path that
    HotpotQA tests, so a model that loses its multi-step composition
    ability still passes the v1 guard.

v2.1 (2026-05-08) replaces the single-probe gate with three per-probe
accuracy measurements aligned to the v2.1 training panel
(FEVER + TriviaQA + CommonsenseQA):

  * MMLU (4-choice MCQ retention; same as v1)
  * TriviaQA test split (open-text factoid retention)
  * CommonsenseQA validation split (5-choice MCQ commonsense retention)

The HotpotQA probe from v2 was retired alongside HotpotQA's removal from
the training panel (precondition violation, see branch_C_log 2026-05-08).
Its registration is kept (just no longer in DEFAULT_PROBES) for
back-compat with legacy ablations + as observability if a future
trajectory re-introduces multi-hop training.

The forgetting guard is now: **abort iff ANY probe drops by more
than (1 − tolerance) from its pristine cycle-0 value**. Any one
failure mode short-circuits the cycle; the multi-probe AND-of-OK
makes the guard strictly more conservative than v1's single-probe
gate.

Public surface
--------------
* :func:`run_retention_probes` — runs each probe in the registry,
  returns a ``{probe_name: accuracy}`` dict. Probes that fail to load
  (no internet, no HF cache) emit NaN so the dict shape is stable.
* :data:`PROBE_REGISTRY` — maps probe name to its runner function.
  Tests can register their own probe via the public registry to
  exercise the orchestration without touching real datasets.
* :func:`retention_ratios` — given a pristine baseline dict and a
  current dict, compute per-probe ratios.
* :func:`any_probe_below_tolerance` — boolean halt criterion: True iff
  any probe's ratio < tolerance.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# Probe registry                                                                #
# ---------------------------------------------------------------------------- #

ProbeRunner = Callable[[Any, Any, int], float]
"""Signature: (model, tokenizer, n) -> accuracy in [0, 1]."""

PROBE_REGISTRY: Dict[str, ProbeRunner] = {}


def register_probe(name: str, runner: ProbeRunner) -> None:
    """Register a probe runner under ``name``. Overwrites silently."""
    PROBE_REGISTRY[name] = runner


def unregister_probe(name: str) -> None:
    """Remove a probe from the registry (used by tests for cleanup)."""
    PROBE_REGISTRY.pop(name, None)


# ---------------------------------------------------------------------------- #
# Built-in probes                                                               #
# ---------------------------------------------------------------------------- #

def _mmlu_probe(model: Any, tokenizer: Any, n: int) -> float:
    """4-choice MMLU validation accuracy. Mirrors v1's
    SelfImprovementLoop._mmlu_score but lives in the retention-probe
    namespace. Returns NaN on dataset load failure."""
    try:
        from datasets import load_dataset
        import torch
    except ImportError as exc:
        logger.warning("mmlu_probe: imports unavailable (%s) → NaN.", exc)
        return float("nan")
    try:
        ds = load_dataset("cais/mmlu", "all", split="validation")
        ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    except Exception as exc:
        logger.warning("mmlu_probe: load failed (%s) → NaN.", exc)
        return float("nan")

    correct = 0
    total = 0
    labels = ["A", "B", "C", "D"]

    model.eval()
    with torch.no_grad():
        for item in ds:
            item_d: dict = item  # type: ignore[assignment]
            question = str(item_d.get("question", ""))
            choices = item_d.get("choices", [])
            answer_idx = int(item_d.get("answer", -1))
            if not question or not choices or not (0 <= answer_idx < 4):
                continue
            choice_str = "\n".join(
                f"{labels[i]}. {c}" for i, c in enumerate(choices) if i < 4
            )
            prompt_text = (
                f"<|im_start|>user\n"
                f"Question: {question}\nChoices:\n{choice_str}\n"
                f"Answer with just the letter (A, B, C, or D).\n"
                f"<|im_end|>\n<|im_start|>assistant\n"
            )
            try:
                enc = tokenizer(prompt_text, return_tensors="pt").to(
                    next(model.parameters()).device
                )
                out = model.generate(
                    **enc, max_new_tokens=8, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or 0,
                )
                gen = tokenizer.decode(
                    out[0][enc["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
            except Exception:
                continue
            for ch in gen.strip()[:8]:
                if ch.upper() in labels:
                    if labels.index(ch.upper()) == answer_idx:
                        correct += 1
                    total += 1
                    break
    if total == 0:
        return float("nan")
    return correct / total


def _triviaqa_test_probe(model: Any, tokenizer: Any, n: int) -> float:
    """Open-text factoid retention via TriviaQA test split. Returns NaN
    on dataset load failure. Scoring: any-match exact-match (gold has
    multiple aliases per question)."""
    try:
        from datasets import load_dataset
        import torch
    except ImportError as exc:
        logger.warning("triviaqa_probe: imports unavailable (%s) → NaN.", exc)
        return float("nan")
    try:
        ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    except Exception as exc:
        logger.warning("triviaqa_probe: load failed (%s) → NaN.", exc)
        return float("nan")

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for item in ds:
            item_d: dict = item  # type: ignore[assignment]
            question = str(item_d.get("question", ""))
            answer = item_d.get("answer", {})
            gold_aliases = list(answer.get("aliases", [])) if isinstance(answer, dict) else []
            if not question or not gold_aliases:
                continue
            prompt_text = (
                f"<|im_start|>user\n"
                f"Question: {question}\n"
                f"Give a SHORT answer (1-5 words preferred).\n"
                f"<|im_end|>\n<|im_start|>assistant\n"
            )
            try:
                enc = tokenizer(prompt_text, return_tensors="pt").to(
                    next(model.parameters()).device
                )
                out = model.generate(
                    **enc, max_new_tokens=32, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or 0,
                )
                gen = tokenizer.decode(
                    out[0][enc["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip().lower()
            except Exception:
                continue
            total += 1
            if any(a.lower() in gen for a in gold_aliases):
                correct += 1
    if total == 0:
        return float("nan")
    return correct / total


def _hotpotqa_test_probe(model: Any, tokenizer: Any, n: int) -> float:
    """Multi-hop composition retention via HotpotQA validation split.
    Returns NaN on dataset load failure. Scoring: substring match
    against the gold single answer."""
    try:
        from datasets import load_dataset
        import torch
    except ImportError as exc:
        logger.warning("hotpotqa_probe: imports unavailable (%s) → NaN.", exc)
        return float("nan")
    try:
        ds = load_dataset("hotpot_qa", "distractor", split="validation")
        ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    except Exception as exc:
        logger.warning("hotpotqa_probe: load failed (%s) → NaN.", exc)
        return float("nan")

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for item in ds:
            item_d: dict = item  # type: ignore[assignment]
            question = str(item_d.get("question", ""))
            gold = str(item_d.get("answer", "")).strip().lower()
            if not question or not gold:
                continue
            prompt_text = (
                f"<|im_start|>user\n"
                f"Question: {question}\n"
                f"The question requires combining information across multiple "
                f"facts; reason through the steps in order, then give a SHORT "
                f"final answer.\n"
                f"<|im_end|>\n<|im_start|>assistant\n"
            )
            try:
                enc = tokenizer(prompt_text, return_tensors="pt").to(
                    next(model.parameters()).device
                )
                out = model.generate(
                    **enc, max_new_tokens=64, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or 0,
                )
                gen = tokenizer.decode(
                    out[0][enc["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip().lower()
            except Exception:
                continue
            total += 1
            if gold in gen:
                correct += 1
    if total == 0:
        return float("nan")
    return correct / total


def _commonsense_qa_test_probe(model: Any, tokenizer: Any, n: int) -> float:
    """5-choice MCQ retention via CommonsenseQA validation split. Returns
    NaN on dataset load failure. Scoring: extract first letter A-E from the
    generation and compare to the gold answer key.

    v2.1 (2026-05-08): added when CSQA replaced HotpotQA's slot in the
    retention probe panel after HotpotQA was removed from training.
    """
    try:
        from datasets import load_dataset
        import re
        import torch
    except ImportError as exc:
        logger.warning("commonsense_qa_probe: imports unavailable (%s) → NaN.", exc)
        return float("nan")
    try:
        ds = load_dataset("tau/commonsense_qa", split="validation")
        ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    except Exception as exc:
        logger.warning("commonsense_qa_probe: load failed (%s) → NaN.", exc)
        return float("nan")

    correct = 0
    total = 0
    model.eval()
    _ans_re = re.compile(r"\b([A-E])\b")
    with torch.no_grad():
        for item in ds:
            item_d: dict = item  # type: ignore[assignment]
            question = str(item_d.get("question", ""))
            choices = item_d.get("choices") or {}
            labels = list(choices.get("label", [])) if isinstance(choices, dict) else []
            texts = list(choices.get("text", [])) if isinstance(choices, dict) else []
            gold_key = str(item_d.get("answerKey", "")).strip().upper()
            if not question or not gold_key or len(labels) != len(texts) or not labels:
                continue
            options_block = "\n".join(f"({lbl}) {txt}" for lbl, txt in zip(labels, texts))
            prompt_text = (
                f"<|im_start|>user\n"
                f"Pick the best answer letter (A-E) for the question.\n"
                f"Question: {question}\n"
                f"{options_block}\n"
                f"Reply with just the single letter.\n"
                f"<|im_end|>\n<|im_start|>assistant\n"
            )
            try:
                enc = tokenizer(prompt_text, return_tensors="pt").to(
                    next(model.parameters()).device
                )
                out = model.generate(
                    **enc, max_new_tokens=8, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or 0,
                )
                gen = tokenizer.decode(
                    out[0][enc["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip().upper()
            except Exception:
                continue
            total += 1
            m = _ans_re.search(gen)
            if m and m.group(1) == gold_key:
                correct += 1
    if total == 0:
        return float("nan")
    return correct / total


# Register the built-in probes at import time.
register_probe("mmlu", _mmlu_probe)
register_probe("triviaqa_test", _triviaqa_test_probe)
register_probe("hotpotqa_test", _hotpotqa_test_probe)  # kept for back-compat / observability
register_probe("commonsense_qa_test", _commonsense_qa_test_probe)


# ---------------------------------------------------------------------------- #
# Orchestration                                                                  #
# ---------------------------------------------------------------------------- #

# v2.1 (2026-05-08): default panel aligned with v2.1 training panel
# (FEVER + TriviaQA + CSQA). HotpotQA probe still registered above but
# excluded from defaults — re-add to retention_probes config if the
# trajectory ever re-introduces multi-hop training.
DEFAULT_PROBES: List[str] = ["mmlu", "triviaqa_test", "commonsense_qa_test"]


def run_retention_probes(
    model: Any,
    tokenizer: Any,
    *,
    probes: Optional[List[str]] = None,
    n_per_probe: int = 200,
) -> Dict[str, float]:
    """Run every probe in ``probes`` and return a dict of accuracies.

    Probes that are not in PROBE_REGISTRY are skipped with a warning
    (the dict will lack that key). Built-in probes return NaN on
    dataset load failure rather than raising — NaN propagates cleanly
    through the retention-ratio computation as ``inactive guard``.
    """
    chosen = list(probes) if probes is not None else list(DEFAULT_PROBES)
    out: Dict[str, float] = {}
    for name in chosen:
        runner = PROBE_REGISTRY.get(name)
        if runner is None:
            logger.warning(
                "run_retention_probes: probe '%s' not registered; skipping.",
                name,
            )
            continue
        try:
            acc = float(runner(model, tokenizer, n_per_probe))
        except Exception as exc:
            logger.warning(
                "run_retention_probes: probe '%s' raised (%s); recording NaN.",
                name, exc,
            )
            acc = float("nan")
        out[name] = acc
    return out


def retention_ratios(
    pristine: Dict[str, float],
    current: Dict[str, float],
) -> Dict[str, float]:
    """Compute per-probe ratios current[p] / pristine[p].

    NaN propagates: if either side is NaN or pristine ≤ 0, the ratio
    for that probe is NaN (callers should treat NaN as "guard inactive
    for this probe", not "abort").
    """
    out: Dict[str, float] = {}
    for name, base in pristine.items():
        if (
            base is None
            or math.isnan(float(base))
            or float(base) <= 1e-6
        ):
            out[name] = float("nan")
            continue
        cur = current.get(name, float("nan"))
        if cur is None or math.isnan(float(cur)):
            out[name] = float("nan")
            continue
        out[name] = float(cur) / float(base)
    return out


def any_probe_below_tolerance(
    ratios: Dict[str, float],
    tolerance: float,
) -> bool:
    """Halt criterion: True iff at least one finite ratio is < tolerance.

    NaN ratios (load failures) are treated as "inactive" — they neither
    trigger nor block the abort. If EVERY ratio is NaN, returns False
    (no guard is active, so no abort signal).
    """
    for name, r in ratios.items():
        if r is None:
            continue
        try:
            rv = float(r)
        except (TypeError, ValueError):
            continue
        if math.isnan(rv):
            continue
        if rv < float(tolerance):
            return True
    return False


def worst_probe(ratios: Dict[str, float]) -> Optional[str]:
    """Return the name of the probe with the lowest finite ratio, or
    None when every ratio is NaN. Used for log diagnostics."""
    candidates = [
        (name, float(r))
        for name, r in ratios.items()
        if r is not None and not math.isnan(float(r))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[1])
    return candidates[0][0]
