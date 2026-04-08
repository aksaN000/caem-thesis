"""
scripts/run_ablation.py
========================
CAEM Ablation Studies + 6 Baseline Comparisons

Runs all comparison experiments for Chapter 5, Section 5.4 (Ablation Analysis).

Six baselines (from benchmarks-and-baselines.md):
  A0 -- Zero-shot       : Flan-T5-Large, no system at all
  A1 -- CoT             : Chain-of-thought prompting, no memory
  A2 -- RAG-only        : Wikipedia retrieval, no memory or verification
  A3 -- Self-consistency: SC majority vote, no memory or fine-tuning
  A4 -- Vanilla FT      : Fine-tuned on unverified data (no verification gate)
  A5 -- Memory-only     : Episodic memory without self-improvement loop

Seven ablation variants (what happens when you remove one CAEM mechanism):
  AB1 -- No memory           : Full CAEM pipeline but empty episodic store (always Tier 3)
  AB2 -- No verification     : Skip MultiLayerVerifier (store everything, u_stored=0.5)
  AB3 -- No CoT              : Fine-tune on short answer strings, not reasoning chains
                               (requires separate checkpoint -- skipped if not found)
  AB4 -- No re-verification  : Retroactive re-verification disabled between cycles
                               (requires separate checkpoint -- skipped if not found)
  AB5 -- NLI only            : Verification uses only NLI layer (no SC, no SE)
  AB6 -- No OR-condition     : Safety veto (u_pre < 0.60 → Tier 3) removed
  AB7 -- No semantic entropy : Verification uses NLI + SC only (SE layer removed)

Each condition runs on all 4 benchmarks against the Cycle 3 state.
Results are compared against full CAEM (loaded from outputs/all_cycle_results.json).

Fixes applied (audit 2025-04)
------------------------------
  FIX-1  eval_baseline(): TruthfulQA now uses ROUGE-L > 0.15 threshold (matches
         eval/harness.py L267-269); StrategyQA uses extract_strategyqa_label()
         (matches harness.py L272-275).
  FIX-2  _clone_pipeline(): creates fresh EpisodicMemoryStore instead of sharing
         the base pipeline's memory store (prevents cross-contamination).
  FIX-3  VanillaFinetuneBaseline: loads unverified-data checkpoint if present;
         warns and uses Cycle 3 weights as fallback.
  FIX-4  MemoryOnlyBaseline: uses Cycle 0 checkpoint (base weights + no SIL);
         warns and uses Cycle 3 weights as fallback.
  FIX-5  AB3/AB4: skip gracefully with a clear warning when checkpoint is absent
         (previously silently returned base_pipeline, producing identical results
         to full CAEM and corrupting the ablation table).
  FIX-6  Added AB5 (NLI-only), AB6 (no OR-condition), AB7 (no SE) ablations.
  FIX-7  MMLU retention measured per ablation (not only for the full pipeline).
  FIX-8  Default output_dir changed to outputs/ablation_results (matches
         NEXT_SESSION_PLAN.md expectation).

Thesis reference
----------------
  §5.4 Ablation analysis
  §5.5 Comparison with baselines
  benchmarks-and-baselines.md -- baseline descriptions

Usage
-----
  python -m scripts.run_ablation \\
      --caem_results outputs/all_cycle_results.json \\
      --cycle3_checkpoint outputs/cycle_3 \\
      --output_dir outputs/ablation_results \\
      --n_questions 500

  # With optional separate checkpoints:
  python -m scripts.run_ablation \\
      --cycle3_checkpoint outputs/cycle_3 \\
      --cycle0_checkpoint outputs/cycle_0 \\
      --unverified_checkpoint outputs/ablation/vanilla_ft \\
      --no_cot_checkpoint outputs/ablation/no_cot \\
      --no_reverif_checkpoint outputs/ablation/no_reverif

  # Smoke test (CPU, synthetic data):
  python -m scripts.run_ablation --smoke_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, cast

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Baseline implementations
# -----------------------------------------------------------------------------

class ZeroShotBaseline:
    """A0: Plain Flan-T5-Large with no system (floor baseline)."""

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def answer(self, question: str) -> str:
        inputs = self.tokenizer(
            question, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with __import__("torch").no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class CoTBaseline:
    """A1: Chain-of-thought prompting -- 'Let's think step by step' prefix."""

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def answer(self, question: str) -> str:
        prompt = f"Question: {question}\nThink step by step:"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with __import__("torch").no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class RAGOnlyBaseline:
    """A2: Retrieval-augmented generation without episodic memory or verification."""

    def __init__(self, model, tokenizer, encoder, passage_store, config, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder          # QueryEncoder -- required by TierThreeRAG
        self.passage_store = passage_store
        self.config = config
        self.device = device

    def answer(self, question: str) -> str:
        if self.passage_store is None:
            # Fallback: zero-shot if no passage index
            inputs = self.tokenizer(
                question, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            with __import__("torch").no_grad():
                out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
            return self.tokenizer.decode(out[0], skip_special_tokens=True)

        # TierThreeRAG constructor: model, tokenizer, passage_encoder, passage_store, config, device
        from caem.retrieval.rag import TierThreeRAG
        rag = TierThreeRAG(
            model=self.model,
            tokenizer=self.tokenizer,
            passage_encoder=self.encoder,
            passage_store=self.passage_store,
            config=self.config,
            device=self.device,
        )
        return rag.generate(question)


class SelfConsistencyBaseline:
    """A3: Self-consistency majority vote -- no memory or fine-tuning.

    Generates N=10 answers and returns the most frequent one.
    This isolates the memory + routing contribution over plain SC.
    """

    N = 10

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def answer(self, question: str) -> str:
        import torch
        inputs = self.tokenizer(
            question, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        answers: List[str] = []
        with torch.no_grad():
            for _ in range(self.N):
                out = self.model.generate(
                    **inputs, max_new_tokens=256,
                    do_sample=True, temperature=1.0,
                )
                answers.append(
                    self.tokenizer.decode(out[0], skip_special_tokens=True)
                )

        # Return majority vote
        from collections import Counter
        return Counter(answers).most_common(1)[0][0]


class VanillaFinetuneBaseline:
    """A4: Fine-tuned on unverified data.

    To isolate the value of CAEM's verification gate:
    we fine-tune on ALL generated answers regardless of u_stored.
    This answers 'does the verified data selection matter?'

    Implementation: loads a checkpoint trained without the verification
    filter (u_stored threshold set to 0.0 -- everything stored).
    Checkpoint path: --unverified_checkpoint (outputs/ablation/vanilla_ft).

    FIX-3: Previously this class was just raw model.generate() (zero-shot),
    which is identical to A0 and defeats the purpose of the baseline.
    Now it loads the unverified checkpoint when available.
    """

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def answer(self, question: str) -> str:
        inputs = self.tokenizer(
            question, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with __import__("torch").no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


class MemoryOnlyBaseline:
    """A5: Memory without self-improvement -- isolates what the fine-tuning loop adds.

    Uses the Cycle 0 model (base Flan-T5-Large weights, no SIL updates) with
    episodic memory routing enabled. Comparing its Cycle 3 EM against full CAEM
    Cycle 3 quantifies the self-improvement loop contribution.

    FIX-4: Previously this was using the Cycle 3 model (post fine-tuning weights),
    which tested 'full CAEM but calling it memory-only'. Now it loads the Cycle 0
    checkpoint (base weights). The memory store is populated normally during Cycle 0
    accumulation -- only the SIL fine-tuning step is absent.

    Checkpoint path: --cycle0_checkpoint (outputs/cycle_0).
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def answer(self, question: str) -> str:
        result = self.pipeline.answer(question)
        return result.answer


# -----------------------------------------------------------------------------
# CAEM ablation variants
# -----------------------------------------------------------------------------

def build_no_memory_pipeline(base_pipeline):
    """AB1: Full CAEM without episodic memory.

    Achieved by replacing the memory store with one that always returns
    empty search results (similarity 0.0), forcing every query to Tier 3.
    The model still runs Tier 3 RAG, so this isolates the memory contribution.
    """
    from caem.memory.store import EpisodicMemoryStore

    class EmptyMemoryStore(EpisodicMemoryStore):
        def search(self, embedding, k=1):
            return []
        def search_with_ids(self, embedding, k=1):
            return []

    from caem.pipeline import CAEMPipeline
    new_pipeline = CAEMPipeline(
        model=base_pipeline.model,
        tokenizer=base_pipeline.tokenizer,
        encoder=base_pipeline.encoder,
        nli_model=getattr(base_pipeline.verifier, "nli_model", None),
        nli_tokenizer=getattr(base_pipeline.verifier, "nli_tokenizer", None),
        passage_store=base_pipeline.rag.passage_store
            if hasattr(base_pipeline, "rag") and base_pipeline.rag else None,
        config=base_pipeline.config,
        memory_store=EmptyMemoryStore(base_pipeline.config),
        device=base_pipeline.device,
    )
    return new_pipeline


def build_no_verification_pipeline(base_pipeline):
    """AB2: Full CAEM without verification (store everything at u_stored=0.5).

    Achieved by overriding MultiLayerVerifier to always return a pass
    with a fixed u_stored of 0.5. This shows the value of the verification
    quality gate -- without it, memory fills with unverified (potentially
    wrong) answers.
    """
    from caem.memory.entry import StoredConfidence
    from caem.verification.verifier import MultiLayerVerifier

    class PassthroughVerifier(MultiLayerVerifier):
        def verify(self, query: str, answer: str, input_ids=None):
            """Always pass at u_stored=0.5 (neutral confidence).

            StoredConfidence fields: p_entail, s_avg, h_norm, u_stored.
            """
            from caem.memory.entry import StoredConfidence
            return StoredConfidence(
                p_entail=0.5,
                s_avg=0.5,
                h_norm=0.5,
                u_stored=0.5,
            )

    new_pipeline = _clone_pipeline(base_pipeline)
    new_pipeline.verifier = PassthroughVerifier(
        model=base_pipeline.model,
        tokenizer=base_pipeline.tokenizer,
        sbert_encoder=base_pipeline.encoder,
        config=new_pipeline.config,
        device=base_pipeline.device,
    )
    return new_pipeline


def build_no_cot_pipeline(base_pipeline, no_cot_ckpt_dir: Optional[str] = None):
    """AB3: Fine-tuned on short answer strings, not reasoning chains.

    This variant uses a model fine-tuned with QAPair(answer=entry.answer)
    instead of QAPair(answer=entry.reasoning_chain).

    FIX-5: Previously returned base_pipeline silently (giving identical CAEM
    results). Now returns None if the checkpoint is missing, so the ablation
    is cleanly skipped in the results table.

    To generate the checkpoint:
      re-run run_experiment.py with --no_cot_supervision flag
      and --output_dir outputs/ablation/no_cot

    Parameters
    ----------
    no_cot_ckpt_dir : str or None -- path to no-CoT checkpoint directory
    """
    import torch
    from transformers import T5ForConditionalGeneration

    if no_cot_ckpt_dir is None:
        logger.warning(
            "AB3 (no-CoT): --no_cot_checkpoint not provided. "
            "Skipping this ablation. To run it: re-train with "
            "--no_cot_supervision and pass --no_cot_checkpoint <path>."
        )
        return None

    ckpt_path = Path(no_cot_ckpt_dir) / "model.pt"
    if not ckpt_path.exists():
        logger.warning(
            "AB3 (no-CoT): checkpoint not found at %s. "
            "Skipping this ablation.", ckpt_path
        )
        return None

    from caem.pipeline import CAEMPipeline
    import copy
    no_cot_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    no_cot_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    if hasattr(base_pipeline, "_dtype_flag"):
        pass  # dtype handled by caller
    no_cot_model = no_cot_model.to(base_pipeline.device).eval()

    from caem.memory.store import EpisodicMemoryStore
    new_pipeline = CAEMPipeline(
        model=no_cot_model,
        tokenizer=base_pipeline.tokenizer,
        encoder=base_pipeline.encoder,
        nli_model=getattr(base_pipeline.verifier, "nli_model", None),
        nli_tokenizer=getattr(base_pipeline.verifier, "nli_tokenizer", None),
        passage_store=base_pipeline.rag.passage_store
            if hasattr(base_pipeline, "rag") and base_pipeline.rag else None,
        config=base_pipeline.config,
        memory_store=EpisodicMemoryStore(base_pipeline.config),
        device=base_pipeline.device,
    )
    logger.info("AB3 (no-CoT): loaded checkpoint from %s.", ckpt_path)
    return new_pipeline


def build_no_reverification_pipeline(base_pipeline, no_reverif_ckpt_dir: Optional[str] = None):
    """AB4: Full pipeline but Retroactive Re-verification is disabled between cycles.

    This variant uses a checkpoint from a run where between-cycle memory scrubbing
    was disabled, so errors that slip through initially are never cleaned up.
    This isolates the value of the retroactive re-verification stage.

    FIX-5: Previously returned base_pipeline silently (identical to full CAEM).
    Now returns None if the checkpoint is missing.

    To generate the checkpoint:
      re-run run_experiment.py with --disable_reverification flag
      and --output_dir outputs/ablation/no_reverif

    Parameters
    ----------
    no_reverif_ckpt_dir : str or None -- path to no-reverification checkpoint directory
    """
    import torch
    from transformers import T5ForConditionalGeneration

    if no_reverif_ckpt_dir is None:
        logger.warning(
            "AB4 (no-reverification): --no_reverif_checkpoint not provided. "
            "Skipping this ablation. To run it: re-train with "
            "--disable_reverification and pass --no_reverif_checkpoint <path>."
        )
        return None

    ckpt_path = Path(no_reverif_ckpt_dir) / "model.pt"
    if not ckpt_path.exists():
        logger.warning(
            "AB4 (no-reverification): checkpoint not found at %s. "
            "Skipping this ablation.", ckpt_path
        )
        return None

    from caem.pipeline import CAEMPipeline
    from caem.memory.store import EpisodicMemoryStore
    no_reverif_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    no_reverif_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    no_reverif_model = no_reverif_model.to(base_pipeline.device).eval()

    new_pipeline = CAEMPipeline(
        model=no_reverif_model,
        tokenizer=base_pipeline.tokenizer,
        encoder=base_pipeline.encoder,
        nli_model=getattr(base_pipeline.verifier, "nli_model", None),
        nli_tokenizer=getattr(base_pipeline.verifier, "nli_tokenizer", None),
        passage_store=base_pipeline.rag.passage_store
            if hasattr(base_pipeline, "rag") and base_pipeline.rag else None,
        config=base_pipeline.config,
        memory_store=EpisodicMemoryStore(base_pipeline.config),
        device=base_pipeline.device,
    )
    logger.info("AB4 (no-reverification): loaded checkpoint from %s.", ckpt_path)
    return new_pipeline


def build_nli_only_pipeline(base_pipeline):
    """AB5: Verification uses NLI layer only (no SC, no SE).

    Overrides MultiLayerVerifier to compute u_stored from NLI entailment
    probability alone, bypassing self-consistency and semantic entropy.
    Isolates the contribution of the multi-signal stacking design.
    """
    from caem.verification.verifier import MultiLayerVerifier
    from caem.memory.entry import StoredConfidence

    class NLIOnlyVerifier(MultiLayerVerifier):
        def verify(self, query: str, answer: str, input_ids=None):
            """Compute u_stored from NLI entailment only (no SC, no SE)."""
            if input_ids is None:
                input_ids = self._tokenize(query)["input_ids"]
            p_entail = self._compute_p_entail(query, answer, input_ids)
            # u_stored = p_entail directly (single signal)
            u_stored = float(min(1.0, max(0.0, p_entail)))
            return StoredConfidence(
                p_entail=p_entail,
                s_avg=0.5,   # SC not computed
                h_norm=0.5,  # SE not computed
                u_stored=u_stored,
            )

    new_pipeline = _clone_pipeline(base_pipeline)
    new_pipeline.verifier = NLIOnlyVerifier(
        model=base_pipeline.model,
        tokenizer=base_pipeline.tokenizer,
        sbert_encoder=base_pipeline.encoder,
        # Pass NLI model -- without this _compute_p_entail silently returns 0.5
        # (verifier.py L205-207 fallback), making AB5 meaningless.
        nli_model=base_pipeline.verifier.nli_model,
        nli_tokenizer=base_pipeline.verifier.nli_tokenizer,
        config=new_pipeline.config,
        device=base_pipeline.device,
    )
    return new_pipeline


def build_no_or_condition_pipeline(base_pipeline):
    """AB6: Safety veto (u_pre < threshold → force Tier 3) removed.

    The OR-condition in the router says: even if similarity >= tier1_threshold,
    if u_pre < safety_u_pre_min (0.60), route to Tier 3 instead.
    This ablation disables that veto to isolate its contribution.

    Achieved by setting safety_u_pre_min to -1.0 in this variant's config,
    which makes the OR-condition unreachable for normal u_pre in [0, 1].
    """
    new_pipeline = _clone_pipeline(base_pipeline)
    new_pipeline.config.safety_u_pre_min = -1.0
    logger.info("AB6 (no OR-condition): set safety_u_pre_min=-1.0 for this variant.")
    return new_pipeline


def build_no_se_pipeline(base_pipeline):
    """AB7: Verification uses NLI + SC only (semantic entropy layer removed).

    Overrides MultiLayerVerifier to skip the SE computation, falling back
    to NLI + SC for the u_stored signal. Isolates the SE contribution.
    """
    from caem.verification.verifier import MultiLayerVerifier
    from caem.memory.entry import StoredConfidence

    class NoSEVerifier(MultiLayerVerifier):
        def verify(self, query: str, answer: str, input_ids=None):
            """Compute u_stored from NLI + SC only (SE bypassed)."""
            if input_ids is None:
                input_ids = self._tokenize(query)["input_ids"]
            p_entail = self._compute_p_entail(query, answer, input_ids)
            s_avg = self._compute_s_avg(query, answer, input_ids)
            # u_stored = weighted average of NLI and SC only
            # Match the two-signal weighting from config
            cfg = self.config
            nli_w = cfg.u_stored_weight_nli
            sc_w  = cfg.u_stored_weight_sc
            total = nli_w + sc_w
            u_stored = (nli_w * p_entail + sc_w * s_avg) / total if total > 0 else 0.5
            u_stored = float(min(1.0, max(0.0, u_stored)))
            return StoredConfidence(
                p_entail=p_entail,
                s_avg=s_avg,
                h_norm=0.5,  # SE not computed
                u_stored=u_stored,
            )

    new_pipeline = _clone_pipeline(base_pipeline)
    new_pipeline.verifier = NoSEVerifier(
        model=base_pipeline.model,
        tokenizer=base_pipeline.tokenizer,
        sbert_encoder=base_pipeline.encoder,
        # Pass NLI model -- without this _compute_p_entail silently returns 0.5
        # (verifier.py L205-207 fallback), corrupting the NLI+SC signal for AB7.
        nli_model=base_pipeline.verifier.nli_model,
        nli_tokenizer=base_pipeline.verifier.nli_tokenizer,
        config=new_pipeline.config,
        device=base_pipeline.device,
    )
    return new_pipeline


def _clone_pipeline(base_pipeline):
    """Create a copy of the pipeline sharing model weights but with a FRESH memory store.

    FIX-2: The original version shared base_pipeline.memory_store, meaning all
    cloned pipelines read from and write to the same store. This caused
    cross-contamination between ablation conditions. Each ablation now gets an
    independent empty store so only its own accumulated episodes affect routing.
    """
    import copy
    from caem.pipeline import CAEMPipeline
    from caem.memory.store import EpisodicMemoryStore  # fresh store, not shared

    cloned_config = copy.deepcopy(base_pipeline.config)

    return CAEMPipeline(
        model=base_pipeline.model,
        tokenizer=base_pipeline.tokenizer,
        encoder=base_pipeline.encoder,
        nli_model=getattr(base_pipeline.verifier, "nli_model", None),
        nli_tokenizer=getattr(base_pipeline.verifier, "nli_tokenizer", None),
        passage_store=base_pipeline.rag.passage_store
            if hasattr(base_pipeline, "rag") and base_pipeline.rag else None,
        config=cloned_config,
        memory_store=EpisodicMemoryStore(cloned_config),  # FIX-2: fresh store
        device=base_pipeline.device,
    )


# -----------------------------------------------------------------------------
# Generic eval runner for non-CAEM baselines
# -----------------------------------------------------------------------------

def eval_baseline(
    name: str,
    baseline,
    samples: Dict[str, list],
    output_dir: Path,
) -> Dict[str, Dict]:
    """Evaluate a baseline system on all benchmarks.

    Parameters
    ----------
    name     : str -- identifier for filenames (e.g. "zero_shot", "cot")
    baseline : object with .answer(question: str) -> str
    samples  : dict[bm -> list of BenchmarkSample]
    output_dir: Path

    Returns
    -------
    dict[bm -> {em, f1, n}]

    Scoring notes (FIX-1)
    ---------------------
    TruthfulQA : EM = float(ROUGE-L(pred, gold_answers) > 0.15)
                 Matches eval/harness.py L267-269. The old any_match_em() used
                 exact string match which is too strict for natural-language answers.
    StrategyQA : EM uses extract_strategyqa_label(pred) before exact_match.
                 Matches eval/harness.py L272-275. Raw exact_match on free-form
                 output fails to catch "Yes, ..." vs "yes" variants.
    """
    from eval.metrics import (
        exact_match, extract_fever_label,
        extract_strategyqa_label, fever_accuracy, rouge_l, token_f1,
    )

    results: Dict[str, Dict] = {}

    for bm, bm_samples in samples.items():
        em_scores, f1_scores = [], []
        logger.info("  %s | %s -- evaluating %d samples ...", name, bm, len(bm_samples))

        for sample in bm_samples:
            q = sample["question"]
            gold = sample.get("answers", [])
            gold_label = sample.get("gold_label")

            try:
                pred = baseline.answer(q)
            except Exception as exc:
                logger.debug("baseline.answer failed: %s", exc)
                pred = ""

            if bm == "fever":
                pred_label = extract_fever_label(pred)
                ref = gold_label or (gold[0] if gold else "not enough info")
                em = fever_accuracy(pred_label, ref)
                f1 = em
            elif bm == "truthfulqa":
                # FIX-1: ROUGE-L threshold matching eval/harness.py L267-269
                em = float(rouge_l(pred, gold) > 0.15)
                f1 = rouge_l(pred, gold)
            elif bm == "strategyqa":
                # FIX-1: label extraction matching eval/harness.py L272-275
                pred_label = extract_strategyqa_label(pred)
                ref_label  = extract_strategyqa_label(gold[0] if gold else "no")
                em = float(pred_label == ref_label) if pred_label and ref_label else 0.0
                f1 = em
            else:
                ref = gold[0] if gold else ""
                em = exact_match(pred, ref)
                f1 = token_f1(pred, ref)

            em_scores.append(em)
            f1_scores.append(f1)

        n = len(em_scores)
        avg_em = sum(em_scores) / n if n > 0 else 0.0
        avg_f1 = sum(f1_scores) / n if n > 0 else 0.0
        results[bm] = {"em": avg_em, "f1": avg_f1, "n": n}
        logger.info("    %s | %s: EM=%.4f  F1=%.4f", name, bm, avg_em, avg_f1)

    # Save results
    out_path = output_dir / f"baseline_{name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("  Saved -> %s", out_path)

    return results


# -----------------------------------------------------------------------------
# MMLU Retention test (for forgetting measurement)
# FIX-7: Now accepts a label so per-ablation retention can be measured.
# -----------------------------------------------------------------------------

def eval_mmlu_retention(pipeline, n: int = 200, label: str = "pipeline") -> float:
    """Evaluate MMLU accuracy to check for catastrophic forgetting.

    Uses a 200-question sample from MMLU (4-choice multiple choice).
    Target: >= 93% retention vs. the original Flan-T5-Large score.
    This fills the MMLU Retention column in the mechanism evidence table.

    FIX-7: Added `label` parameter so per-ablation MMLU scores can be
    logged and saved distinctly (previously only run on the full pipeline).

    Parameters
    ----------
    pipeline : CAEMPipeline (or MemoryOnlyBaseline pipeline)
    n        : int -- number of MMLU validation questions to use
    label    : str -- identifier for logging (e.g. "full_caem", "ab1_no_memory")

    Returns
    -------
    float -- accuracy on MMLU (0-1), or NaN if dataset unavailable
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="validation")
        ds = ds.select(range(min(n, len(ds))))
    except Exception as exc:
        logger.warning("MMLU load failed (%s). Retention for %s cannot be reported.", exc, label)
        return float("nan")

    correct = 0
    total = 0

    for item in ds:
        row = cast(Mapping[str, Any], item)
        q = str(row.get("question", ""))
        choices = cast(List[str], row.get("choices", []))
        answer_idx = int(row.get("answer", 0))

        prompt = (
            f"Question: {q}\n"
            f"A) {choices[0]}\nB) {choices[1]}\n"
            f"C) {choices[2]}\nD) {choices[3]}\n"
            f"Answer:"
        )

        try:
            result = pipeline.answer(prompt)
            pred = result.answer.strip().upper()
            pred_letter = pred[0] if pred and pred[0] in "ABCD" else "X"
            gold_letter = "ABCD"[answer_idx]
            if pred_letter == gold_letter:
                correct += 1
        except Exception:
            pass
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    logger.info("[%s] MMLU retention: %.4f (%d/%d correct)", label, accuracy, correct, total)
    return accuracy


# -----------------------------------------------------------------------------
# Summary printer
# -----------------------------------------------------------------------------

def print_ablation_table(
    all_results: Dict,
    caem_results: Optional[Dict] = None,
    mmlu_by_condition: Optional[Dict[str, float]] = None,
) -> None:
    """Print comparison table: CAEM vs all baselines and ablation variants."""
    print("\n" + "=" * 95)
    print("ABLATION & BASELINE COMPARISON TABLE  (Chapter 5, Table 2)")
    print("=" * 95)
    print(f"{'Condition':<26} {'HotpotQA':>10} {'TruthfulQA':>12} "
          f"{'FEVER':>8} {'StrategyQA':>12} {'MMLU':>7}")
    print("-" * 95)

    # CAEM rows first
    if caem_results:
        last_cycle_idx = len(caem_results) - 1
        for cycle_num, cycle_res in enumerate(caem_results):
            label = f"CAEM Cycle {cycle_num}"
            mmlu = (mmlu_by_condition or {}).get(f"caem_cycle_{cycle_num}", float("nan"))
            if __import__("math").isnan(mmlu) and cycle_num == last_cycle_idx:
                mmlu = (mmlu_by_condition or {}).get("full_caem", float("nan"))
            row = f"  {label:<24}"
            for bm in ["hotpotqa", "truthfulqa", "fever", "strategyqa"]:
                em = cycle_res.get(bm, {}).get("em", float("nan"))
                row += f"  {em:>8.4f}"
            row += f"  {mmlu:>5.4f}" if not __import__("math").isnan(mmlu) else "     n/a"
            print(row)
        print("-" * 95)

    # Baselines and ablations
    for name, res in all_results.items():
        if res is None:
            label = name.replace("_", " ").title()
            print(f"  {label[:24]:<24}  [SKIPPED -- checkpoint not found]")
            continue
        label = name.replace("_", " ").title()
        mmlu = (mmlu_by_condition or {}).get(name, float("nan"))
        row = f"  {label[:24]:<24}"
        for bm in ["hotpotqa", "truthfulqa", "fever", "strategyqa"]:
            em = res.get(bm, {}).get("em", float("nan"))
            row += f"  {em:>8.4f}"
        row += f"  {mmlu:>5.4f}" if not __import__("math").isnan(mmlu) else "     n/a"
        print(row)

    print("=" * 95)
    print("MMLU target: >= 0.93 retention. Ablations AB3/AB4 require separate checkpoints.")
    print("Interpretation: each CAEM mechanism contributes if removing it (AB1-7)")
    print("hurts more than the respective baseline (A0-A5).\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_ablation(ns: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    output_dir = Path(ns.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from scripts.hardware import print_hardware_summary, apply_memory_flags
    profile = print_hardware_summary()
    apply_memory_flags(profile)

    # -- Load model and dependencies --------------------------------------- #
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    from caem.config import CAEMConfig
    from caem.memory.encoder import QueryEncoder
    from caem.pipeline import CAEMPipeline

    device = profile.device
    config = CAEMConfig()

    if ns.smoke_test:
        from eval.benchmarks import make_synthetic_samples
        samples = {bm: make_synthetic_samples(bm, n=8)
                   for bm in ["hotpotqa", "truthfulqa", "fever", "strategyqa"]}
    else:
        from eval.benchmarks import (
            load_hotpotqa, load_truthfulqa, load_fever, load_strategyqa
        )
        samples = {
            "hotpotqa":   load_hotpotqa(n=ns.n_questions),
            "truthfulqa": load_truthfulqa(n=ns.n_questions),
            "fever":      load_fever(n=ns.n_questions),
            "strategyqa": load_strategyqa(n=ns.n_questions),
        }

    logger.info("Loading Flan-T5-Large (Cycle 3 weights) ...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

    # --- Cycle 3 model (used by A3/A4/VanillaFT and ablation variants) --- #
    model_c3 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    cycle3_ckpt = Path(ns.cycle3_checkpoint)
    if (cycle3_ckpt / "model.pt").exists():
        logger.info("Loading Cycle 3 checkpoint from %s ...", cycle3_ckpt)
        model_c3.load_state_dict(torch.load(cycle3_ckpt / "model.pt", map_location="cpu"))
        logger.info("Cycle 3 model loaded.")
    else:
        logger.warning(
            "No Cycle 3 checkpoint found at %s. "
            "Ablations requiring fine-tuned weights will use base Flan-T5-Large.",
            cycle3_ckpt,
        )

    if profile.use_fp16:
        model_c3 = model_c3.half()
    if profile.use_bf16:
        model_c3 = model_c3.bfloat16()
    model_c3 = cast(Any, model_c3).to(torch.device(device)).eval()

    # --- Cycle 0 / base model (used by A5 MemoryOnly, A0/A1/A2/A3 baselines) --- #
    # FIX-4: A5 (MemoryOnlyBaseline) must use base weights, not Cycle 3 weights.
    # Load Cycle 0 checkpoint if provided; otherwise fall back to base Flan-T5-Large.
    cycle0_ckpt = Path(ns.cycle0_checkpoint)
    model_base = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    if (cycle0_ckpt / "model.pt").exists():
        logger.info("Loading Cycle 0 checkpoint (base weights) from %s ...", cycle0_ckpt)
        model_base.load_state_dict(torch.load(cycle0_ckpt / "model.pt", map_location="cpu"))
    else:
        logger.info(
            "Cycle 0 checkpoint not found at %s -- using raw Flan-T5-Large weights for A0-A3/A5.",
            cycle0_ckpt,
        )
    if profile.use_fp16:
        model_base = model_base.half()
    if profile.use_bf16:
        model_base = model_base.bfloat16()
    model_base = cast(Any, model_base).to(torch.device(device)).eval()

    # --- Unverified fine-tune model (used by A4 VanillaFT) --- #
    # FIX-3: Load the model trained on ALL data (no verification filter).
    unverif_ckpt = Path(ns.unverified_checkpoint)
    model_unverif = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    if (unverif_ckpt / "model.pt").exists():
        logger.info("Loading unverified-FT checkpoint from %s ...", unverif_ckpt)
        model_unverif.load_state_dict(torch.load(unverif_ckpt / "model.pt", map_location="cpu"))
    else:
        logger.warning(
            "Unverified-FT checkpoint not found at %s. "
            "A4 (VanillaFT) will use Cycle 3 weights as fallback -- "
            "results will be identical to full CAEM. "
            "To fix: re-train with --no_verification_filter.",
            unverif_ckpt,
        )
        model_unverif = model_c3  # fallback: same as Cycle 3
    if profile.use_fp16:
        model_unverif = model_unverif.half()
    if profile.use_bf16:
        model_unverif = model_unverif.bfloat16()
    model_unverif = cast(Any, model_unverif).to(torch.device(device)).eval()

    encoder = QueryEncoder(model_name=config.sbert_model, device=device)

    # Shared NLI + passage store for all CAEM pipeline variants
    nli_model, nli_tokenizer = None, None
    try:
        from transformers import AutoModelForSequenceClassification
        logger.info("Loading RoBERTa-Large-MNLI for ablation pipelines ...")
        nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        nli_model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-large-mnli"
        ).to(device)
        nli_model.eval()
    except Exception as exc:
        logger.warning("NLI model load failed (%s); continuing without NLI.", exc)

    shared_passage_store = None
    passage_index_path = Path(ns.passage_index)
    if passage_index_path.exists():
        try:
            from caem.retrieval.rag import PassageStore
            logger.info("Loading shared passage index from %s ...", passage_index_path)
            shared_passage_store = PassageStore.load(str(passage_index_path))
        except Exception as exc:
            logger.warning(
                "Passage index load failed (%s); Tier 3 will run without retrieval context.",
                exc,
            )
    else:
        logger.warning(
            "Passage index not found at %s -- Tier 3 will run without retrieval context.",
            passage_index_path,
        )

    # Build full CAEM pipeline (Cycle 3 weights, for ablation variants)
    pipeline_c3 = CAEMPipeline(
        model=model_c3, tokenizer=tokenizer, encoder=encoder,
        nli_model=nli_model, nli_tokenizer=nli_tokenizer,
        passage_store=shared_passage_store,
        config=config, device=device,
    )

    # Build Cycle 0 pipeline (base weights, for A5 MemoryOnly)
    pipeline_base = CAEMPipeline(
        model=model_base, tokenizer=tokenizer, encoder=encoder,
        nli_model=nli_model, nli_tokenizer=nli_tokenizer,
        passage_store=shared_passage_store,
        config=config, device=device,
    )

    # -- Run all conditions ----------------------------------------------- #
    logger.info("=" * 60)
    logger.info("ABLATION STUDY -- running baselines A0-A5 + ablations AB1-AB7")
    logger.info("=" * 60)

    all_baseline_results: Dict[str, Optional[Dict]] = {}
    mmlu_by_condition: Dict[str, float] = {}

    # A0 -- Zero-shot (base model, no system)
    logger.info("A0: Zero-shot baseline ...")
    baseline_zs = ZeroShotBaseline(model_base, tokenizer, device)
    all_baseline_results["zero_shot"] = eval_baseline("zero_shot", baseline_zs, samples, output_dir)

    # A1 -- CoT (base model, no memory)
    logger.info("A1: CoT baseline ...")
    baseline_cot = CoTBaseline(model_base, tokenizer, device)
    all_baseline_results["cot"] = eval_baseline("cot", baseline_cot, samples, output_dir)

    # A2 -- RAG-only: load passage store from disk.
    logger.info("A2: RAG-only baseline ...")
    _rag_passage_store = shared_passage_store
    if _rag_passage_store is not None:
        _n_passages = getattr(_rag_passage_store, "size", len(getattr(_rag_passage_store, "passages", [])))
        logger.info("RAG baseline: using shared passage store (%d passages).", _n_passages)
    else:
        logger.warning(
            "RAG baseline: passage store unavailable. "
            "RAG-only ablation will degrade to zero-shot -- "
            "build the index first: python -m scripts.build_passage_index",
        )
    baseline_rag = RAGOnlyBaseline(model_base, tokenizer, encoder, _rag_passage_store, config, device)
    all_baseline_results["rag_only"] = eval_baseline("rag_only", baseline_rag, samples, output_dir)

    # A3 -- Self-consistency (base model, no memory, majority vote)
    logger.info("A3: Self-consistency baseline ...")
    baseline_sc = SelfConsistencyBaseline(model_base, tokenizer, device)
    all_baseline_results["self_consistency"] = eval_baseline(
        "self_consistency", baseline_sc, samples, output_dir
    )

    # A4 -- Vanilla fine-tune (trained on unverified data)
    # FIX-3: Uses dedicated unverified-FT checkpoint, not Cycle 3 model
    logger.info("A4: Vanilla fine-tune baseline ...")
    baseline_vft = VanillaFinetuneBaseline(model_unverif, tokenizer, device)
    all_baseline_results["vanilla_ft"] = eval_baseline("vanilla_ft", baseline_vft, samples, output_dir)

    # A5 -- Memory-only (Cycle 0 model weights, memory routing enabled)
    # FIX-4: Uses Cycle 0 model (base weights, no SIL), not Cycle 3 model
    logger.info("A5: Memory-only baseline ...")
    baseline_mem = MemoryOnlyBaseline(pipeline_base)
    all_baseline_results["memory_only"] = eval_baseline("memory_only", baseline_mem, samples, output_dir)

    # AB1 -- No memory (ablation): forces Tier 3 on every query
    logger.info("AB1: No-memory ablation ...")
    no_mem_pipeline = build_no_memory_pipeline(pipeline_c3)
    from eval.harness import EvalHarness
    harness_nomem = EvalHarness(no_mem_pipeline, output_dir=str(output_dir / "ab1_no_memory"), log_every=50)
    all_baseline_results["ab_no_memory"] = harness_nomem.run_all(samples, cycle=3)
    mmlu_by_condition["ab_no_memory"] = eval_mmlu_retention(no_mem_pipeline, n=200, label="ab1_no_memory")

    # AB2 -- No verification (ablation): stores everything at u_stored=0.5
    logger.info("AB2: No-verification ablation ...")
    no_verif_pipeline = build_no_verification_pipeline(pipeline_c3)
    harness_noverif = EvalHarness(no_verif_pipeline, output_dir=str(output_dir / "ab2_no_verification"), log_every=50)
    all_baseline_results["ab_no_verification"] = harness_noverif.run_all(samples, cycle=3)
    mmlu_by_condition["ab_no_verification"] = eval_mmlu_retention(no_verif_pipeline, n=200, label="ab2_no_verification")

    # AB3 -- No CoT (ablation): requires separately trained checkpoint
    # FIX-5: Returns None (skip gracefully) if checkpoint not found
    logger.info("AB3: No-CoT ablation ...")
    no_cot_pipeline = build_no_cot_pipeline(pipeline_c3, no_cot_ckpt_dir=ns.no_cot_checkpoint)
    if no_cot_pipeline is not None:
        harness_nocot = EvalHarness(no_cot_pipeline, output_dir=str(output_dir / "ab3_no_cot"), log_every=50)
        all_baseline_results["ab_no_cot"] = harness_nocot.run_all(samples, cycle=3)
        mmlu_by_condition["ab_no_cot"] = eval_mmlu_retention(no_cot_pipeline, n=200, label="ab3_no_cot")
    else:
        all_baseline_results["ab_no_cot"] = None
        logger.warning("AB3 skipped -- no checkpoint. Re-train with --no_cot_supervision.")

    # AB4 -- No retroactive re-verification (ablation): requires separately trained checkpoint
    # FIX-5: Returns None (skip gracefully) if checkpoint not found
    logger.info("AB4: No-reverification ablation ...")
    no_reverif_pipeline = build_no_reverification_pipeline(pipeline_c3, no_reverif_ckpt_dir=ns.no_reverif_checkpoint)
    if no_reverif_pipeline is not None:
        harness_noreverif = EvalHarness(no_reverif_pipeline, output_dir=str(output_dir / "ab4_no_reverif"), log_every=50)
        all_baseline_results["ab_no_reverif"] = harness_noreverif.run_all(samples, cycle=3)
        mmlu_by_condition["ab_no_reverif"] = eval_mmlu_retention(no_reverif_pipeline, n=200, label="ab4_no_reverif")
    else:
        all_baseline_results["ab_no_reverif"] = None
        logger.warning("AB4 skipped -- no checkpoint. Re-train with --disable_reverification.")

    # AB5 -- NLI only (ablation): single-signal verification
    logger.info("AB5: NLI-only verification ablation ...")
    nli_only_pipeline = build_nli_only_pipeline(pipeline_c3)
    harness_nlionly = EvalHarness(nli_only_pipeline, output_dir=str(output_dir / "ab5_nli_only"), log_every=50)
    all_baseline_results["ab_nli_only"] = harness_nlionly.run_all(samples, cycle=3)
    mmlu_by_condition["ab_nli_only"] = eval_mmlu_retention(nli_only_pipeline, n=200, label="ab5_nli_only")

    # AB6 -- No OR-condition (ablation): safety veto removed from router
    logger.info("AB6: No-OR-condition ablation ...")
    no_or_pipeline = build_no_or_condition_pipeline(pipeline_c3)
    harness_noor = EvalHarness(no_or_pipeline, output_dir=str(output_dir / "ab6_no_or_condition"), log_every=50)
    all_baseline_results["ab_no_or_condition"] = harness_noor.run_all(samples, cycle=3)
    mmlu_by_condition["ab_no_or_condition"] = eval_mmlu_retention(no_or_pipeline, n=200, label="ab6_no_or_cond")

    # AB7 -- No semantic entropy (ablation): NLI + SC only
    logger.info("AB7: No-semantic-entropy ablation ...")
    no_se_pipeline = build_no_se_pipeline(pipeline_c3)
    harness_nose = EvalHarness(no_se_pipeline, output_dir=str(output_dir / "ab7_no_se"), log_every=50)
    all_baseline_results["ab_no_se"] = harness_nose.run_all(samples, cycle=3)
    mmlu_by_condition["ab_no_se"] = eval_mmlu_retention(no_se_pipeline, n=200, label="ab7_no_se")

    # -- MMLU Retention for full CAEM pipeline ----------------------------- #
    logger.info("Measuring MMLU retention for full CAEM (Cycle 3) ...")
    mmlu_c3 = eval_mmlu_retention(pipeline_c3, n=200, label="full_caem_cycle3")
    mmlu_by_condition["full_caem"] = mmlu_c3
    mmlu_by_condition["caem_cycle_3"] = mmlu_c3
    logger.info("Full CAEM MMLU retention: %.4f (target >= 0.93)", mmlu_c3)

    # -- Load full CAEM results for comparison ------------------------------ #
    caem_results = None
    caem_path = Path(ns.caem_results)
    if caem_path.exists():
        with open(caem_path) as f:
            caem_results = json.load(f)
        logger.info("Loaded CAEM results from %s.", caem_path)
    else:
        logger.warning(
            "CAEM results file not found at %s -- comparison table will be incomplete.",
            caem_path,
        )

    # -- Save ablation summary ----------------------------------------------- #
    summary = {
        "baselines": {k: v for k, v in all_baseline_results.items() if v is not None},
        "skipped_ablations": [k for k, v in all_baseline_results.items() if v is None],
        "mmlu_retention_by_condition": mmlu_by_condition,
        "caem_cycle3_results": caem_results[-1] if caem_results else None,
    }
    out_path = output_dir / "ablation_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Ablation summary saved -> %s", out_path)

    # -- Print comparison table ----------------------------------------------- #
    print_ablation_table(all_baseline_results, caem_results, mmlu_by_condition)
    print(f"\nFull CAEM MMLU Retention: {mmlu_c3:.4f} (target >= 0.93)")
    if summary["skipped_ablations"]:
        print(f"Skipped ablations (checkpoint not found): {summary['skipped_ablations']}")
    logger.info("Ablation study complete. Results in: %s", output_dir)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CAEM Ablation Studies + 6 Baseline Comparisons",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--caem_results", default="outputs/all_cycle_results.json",
                   help="Path to all_cycle_results.json from run_experiment.py.")
    p.add_argument("--cycle3_checkpoint", default="outputs/cycle_3",
                   help="Directory containing the Cycle 3 model checkpoint (model.pt).")
    p.add_argument("--cycle0_checkpoint", default="outputs/cycle_0",
                   help="Directory containing the Cycle 0 model checkpoint (model.pt). "
                        "Used by A5 (MemoryOnly) to ensure base weights, not fine-tuned weights.")
    p.add_argument("--unverified_checkpoint", default="outputs/ablation/vanilla_ft",
                   help="Checkpoint trained on ALL data without verification filter. "
                        "Used by A4 (VanillaFT). "
                        "Generate with: run_experiment.py --no_verification_filter.")
    p.add_argument("--no_cot_checkpoint", default=None,
                   help="Checkpoint trained without CoT supervision. "
                        "Used by AB3. Generate with: run_experiment.py --no_cot_supervision.")
    p.add_argument("--no_reverif_checkpoint", default=None,
                   help="Checkpoint trained without retroactive re-verification. "
                        "Used by AB4. Generate with: run_experiment.py --disable_reverification.")
    p.add_argument("--output_dir", default="outputs/ablation_results",
                   help="Root directory for ablation results.")
    p.add_argument("--n_questions", type=int, default=500,
                   help="Questions per benchmark for ablation eval.")
    p.add_argument("--smoke_test", action="store_true",
                   help="Use synthetic data (no download needed).")
    p.add_argument(
        "--passage_index",
        default="data/passage_index",
        help=(
            "Path to pre-built PassageStore directory (passages.faiss + passages.pkl). "
            "Built by scripts/build_passage_index.py. "
            "Required for the RAG-only baseline to be genuine RAG; "
            "without it the baseline degrades to zero-shot."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ablation(args)