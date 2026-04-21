"""
eval/harness.py
===============
EvalHarness -- runs a benchmark against CAEMPipeline and collects results.

Responsibilities
----------------
1. Iterate over BenchmarkSample dicts from eval/benchmarks.py.
2. Call pipeline.answer(question) for each sample.
3. Compute per-sample metrics (EM, F1, FEVER accuracy) via eval/metrics.py.
4. For FEVER: extract the label from the model's free-form answer.
5. Collect routing stats (tier distribution), storage rate, latency.
6. Produce a per-sample result list and an aggregate summary dict.
7. Save results to JSON (per-run) for later analysis.
8. Optionally log progress to stdout for long-running evaluations.

Output schema
-------------
Per-sample (SampleResult dict):
  id, benchmark, question, prediction, gold_answers, gold_label,
  em, f1, tier, stored, u_stored, latency_ms, escalated,
  # Session 42 nine-signal capture (Phase 5e / Phase 4m.3)
  u_token, u_dropout, u_internal, s_avg, h_norm, p_entail,
  p_ground_max, p_ground_mean, p_ground_atomic, p_contra,
  decision, early_exit_triggered
  -- the twelve verifier fields are None when the Stage-5 verifier did not
  run (Tier 1 hits, verifier errors, or fail_on_error=False fall-throughs).

Aggregate (EvalResult dict -- one per run):
  benchmark, cycle, n, em, f1, hallucination_rate,
  storage_rate, mean_u_stored, mean_latency_ms,
  tier1_frac, tier2_frac, tier3_frac

Usage
-----
>>> harness = EvalHarness(pipeline, output_dir="outputs/eval")
>>> result = harness.run(benchmark="fever", samples=samples, cycle=0)
>>> result["em"]          # e.g. 0.55 (FEVER accuracy at Cycle 0)
>>> result["tier3_frac"]  # e.g. 0.884 (mostly Tier 3 before memory warms up)

The output JSON at outputs/eval/fever_cycle0.json can be fed directly
into the analysis scripts for Chapter 5 figures.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import fields as _dc_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from caem.verification.verifier import UnifiedVerifierOutput
from eval.benchmarks import BenchmarkSample, make_synthetic_samples
from eval.metrics import (
    aggregate,
    any_match_em,
    best_token_f1,
    rouge_l,
    exact_match,
    extract_fever_label,
    extract_strategyqa_label,
    extract_arc_label,
    extract_cot_answer,
    fever_accuracy,
    token_f1,
)

logger = logging.getLogger(__name__)

# Type aliases
SampleResult = Dict[str, Any]
EvalResult = Dict[str, Any]


# ---------------------------------------------------------------------------- #
# Verifier-schema registry (single source of truth)                             #
# ---------------------------------------------------------------------------- #

def _derive_verifier_fields() -> Tuple[str, ...]:
    """Scalar (non-list) fields of :class:`UnifiedVerifierOutput`, minus ``u_stored``.

    Derived at import time from ``dataclasses.fields(UnifiedVerifierOutput)``
    so the per-sample JSON schema and the table builders stay in lockstep
    with the verifier dataclass. When a new scalar signal is added to
    ``UnifiedVerifierOutput`` it appears automatically on every subsequent
    eval run -- no parallel tuple to update, no silent drop-through.

    Exclusions:
      * ``u_stored`` -- already captured at the top level of SampleResult
        via ``pipeline.answer().u_stored``; including it here would
        duplicate the key in ``record.update(verifier_signals)``.
      * List-typed fields (``top_passages``, ``atomic_facts``,
        ``per_atom_entail``) -- identified by ``default_factory is list``;
        not flat scalars, not suitable for a rectangular per-sample JSON.

    Audit MAJOR-H3 / Task #118.
    """
    names: List[str] = []
    for f in _dc_fields(UnifiedVerifierOutput):
        if f.name == "u_stored":
            continue
        # List-typed fields use ``field(default_factory=list)``.
        if f.default_factory is list:   # type: ignore[comparison-overlap]
            continue
        names.append(f.name)
    return tuple(names)


# Module-level constant so both ``EvalHarness`` (here) and
# ``eval.reporting.VERIFIER_FIELDS`` can share the single derivation.
VERIFIER_FIELDS: Tuple[str, ...] = _derive_verifier_fields()


class EvalHarness:
    """Run one benchmark against a CAEMPipeline and collect results.

    Parameters
    ----------
    pipeline : CAEMPipeline
        Fully initialised pipeline (model, memory, RAG all set up).
    output_dir : str or Path or None
        If provided, save JSON results here after each run.
        If None, results are returned but not saved to disk.
    log_every : int
        Log progress every N samples (0 = no progress logging).
    fail_on_error : bool
        If True, re-raise exceptions from pipeline.answer().
        If False (default), record the sample as EM=0 and continue.

    Usage
    -----
    >>> harness = EvalHarness(pipeline, output_dir="outputs/eval")
    >>> samples = load_benchmark("fever", n=500, split="dev")
    >>> result = harness.run("fever", samples, cycle=1)
    """

    def __init__(
        self,
        pipeline,
        output_dir: Optional[str | Path] = None,
        log_every: int = 100,
        fail_on_error: bool = False,
        batch_size: int = 1,
        use_prefetch: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.output_dir = Path(output_dir) if output_dir else None
        self.log_every = log_every
        self.fail_on_error = fail_on_error
        # Goal 5 Level B: when batch_size > 1, wrap the serial pipeline in a
        # BatchPipeline (optionally PrefetchingBatchPipeline) and route
        # `run()` through a batched chunked loop. batch_size == 1 preserves
        # the serial per-sample path bit-for-bit (the test equivalence path).
        self.batch_size = max(1, int(batch_size))
        self.use_prefetch = bool(use_prefetch)
        self._batch_pipeline = None  # lazy; built on first batched run

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_batch_pipeline(self):
        if self._batch_pipeline is not None:
            return self._batch_pipeline
        from caem.pipeline_batch import BatchPipeline
        bp = BatchPipeline(self.pipeline)
        if self.use_prefetch:
            try:
                from caem.pipeline_batch_prefetch import PrefetchingBatchPipeline
                bp = PrefetchingBatchPipeline(bp)
                logger.info("EvalHarness: wrapped pipeline with PrefetchingBatchPipeline (bs=%d).", self.batch_size)
            except Exception as exc:
                logger.warning(
                    "PrefetchingBatchPipeline unavailable (%s); using plain BatchPipeline.", exc,
                )
        else:
            logger.info("EvalHarness: wrapped pipeline with BatchPipeline (bs=%d).", self.batch_size)
        self._batch_pipeline = bp
        return bp

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def run(
        self,
        benchmark: str,
        samples: List[BenchmarkSample],
        cycle: int = 0,
        store_to_memory: bool = False,
    ) -> EvalResult:
        """Evaluate all samples from one benchmark at one cycle.

        Parameters
        ----------
        benchmark : str -- "fever", "triviaqa", "natural_questions", "truthfulqa", "strategyqa", or "arc_challenge"
        samples : list of BenchmarkSample
        cycle : int -- current self-improvement cycle (for filename and metadata)

        Returns
        -------
        EvalResult dict -- aggregate metrics for this run
        """
        benchmark = benchmark.lower()
        logger.info(
            "EvalHarness: starting %s | cycle=%d | n=%d samples",
            benchmark, cycle, len(samples),
        )

        sample_results: List[SampleResult] = []

        if self.batch_size > 1:
            sample_results = self._run_batched(samples, benchmark, store_to_memory, cycle)
        else:
            for i, sample in enumerate(samples):
                if self.log_every > 0 and i % self.log_every == 0:
                    logger.info("  [%s/%s] %s cycle=%d ...", i, len(samples), benchmark, cycle)

                sr = self._run_one(sample, benchmark, store_to_memory)
                sample_results.append(sr)

        # -- Aggregate --------------------------------------------------- #
        em_scores = [sr["em"] for sr in sample_results]
        f1_scores = [sr["f1"] for sr in sample_results]
        tiers = [sr["tier"] for sr in sample_results]
        u_stored_vals = [sr["u_stored"] for sr in sample_results]
        stored_flags = [sr["stored"] for sr in sample_results]
        latencies = [sr["latency_ms"] for sr in sample_results]

        agg = aggregate(
            benchmark=benchmark,
            em_scores=em_scores,
            f1_scores=f1_scores,
            tiers=tiers,
            u_stored_values=u_stored_vals,
            stored_flags=stored_flags,
            latencies_ms=latencies,
        )
        agg["cycle"] = cycle

        logger.info(
            "EvalHarness done: %s | cycle=%d | EM=%.4f | F1=%.4f | "
            "Tier1=%.2f%% Tier2=%.2f%% Tier3=%.2f%% | stored=%.2f%%",
            benchmark, cycle,
            agg["em"], agg["f1"],
            agg["tier1_frac"] * 100,
            agg["tier2_frac"] * 100,
            agg["tier3_frac"] * 100,
            agg["storage_rate"] * 100,
        )

        # -- Save -------------------------------------------------------- #
        full_output = {
            "meta": agg,
            "samples": sample_results,
        }

        if self.output_dir:
            fname = f"{benchmark}_cycle{cycle}.json"
            fpath = self.output_dir / fname
            # Atomic write: temp-then-rename prevents truncated JSON on crash.
            # A corrupted eval JSON would break --resume_from_cycle.
            import tempfile as _tf, os as _os
            with _tf.NamedTemporaryFile(
                mode="w", dir=self.output_dir, suffix=".tmp",
                delete=False, encoding="utf-8"
            ) as tmp:
                json.dump(full_output, tmp, indent=2, ensure_ascii=False)
                tmp_path = tmp.name
            _os.replace(tmp_path, fpath)  # atomic on same filesystem
            logger.info("Results saved -> %s", fpath)

        return agg

    # ------------------------------------------------------------------ #
    # Per-sample evaluation                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_verifier_signals(vout) -> Dict[str, Any]:
        """Flatten a UnifiedVerifierOutput into a sample-record dict.

        Returns a dict with all scalar keys set to None when ``vout`` is None
        (Tier 1 hits never run the verifier; Tier 2/3 may fail). This keeps
        the per-sample JSON schema rectangular so downstream analysis scripts
        can concatenate records into a pandas DataFrame cleanly.

        The field tuple is derived once at import time from
        ``UnifiedVerifierOutput`` (see ``VERIFIER_FIELDS`` at module level) so
        schema additions to the verifier propagate automatically without
        hand-edits here. Audit MAJOR-H3 / Task #118.
        """
        if vout is None:
            return {k: None for k in VERIFIER_FIELDS}
        return {k: getattr(vout, k) for k in VERIFIER_FIELDS}

    def _run_one(self, sample: BenchmarkSample, benchmark: str, store_to_memory: bool = False) -> SampleResult:
        """Run one sample through the pipeline and score it.

        Returns a SampleResult dict. On error, records em=0, f1=0, tier=3.
        The per-sample record includes the full twelve-field verifier capture
        (nine signals + p_contra + decision + early_exit_triggered) when the
        Stage-5 verifier ran, else all twelve fields are None (Tier 1 hits,
        pipeline errors with fail_on_error=False).
        """
        try:
            result = self.pipeline.answer(
                sample["question"],
                store_to_memory=store_to_memory,
                source_benchmark=benchmark,
            )
            return self._record_from_result(sample, benchmark, result, pipeline_error=None)
        except Exception as exc:
            if self.fail_on_error:
                raise
            logger.warning("pipeline.answer() raised for sample %s: %s", sample.get("id"), exc)
            return self._record_from_error(sample, benchmark, exc)

    def _record_from_result(
        self,
        sample: BenchmarkSample,
        benchmark: str,
        result,
        pipeline_error: Optional[str] = None,
    ) -> SampleResult:
        """Build a SampleResult from a PipelineResult. Used by both serial and batched paths."""
        question = sample["question"]
        gold_answers = sample["answers"]
        gold_label = sample.get("gold_label")

        prediction = result.answer
        display_answer = getattr(result, "display_answer", result.answer)
        verifier_signals = self._extract_verifier_signals(result.verifier_output)
        em, f1 = self._score(prediction, gold_answers, gold_label, benchmark)

        record: SampleResult = {
            "id": sample.get("id", ""),
            "benchmark": benchmark,
            "question": question,
            "prediction": prediction,
            "display_answer": display_answer,
            "gold_answers": gold_answers,
            "gold_label": gold_label,
            "em": em,
            "f1": f1,
            "tier": result.tier,
            "stored": result.stored,
            "u_stored": result.u_stored,
            "latency_ms": result.latency_ms,
            "escalated": result.escalated,
            "pipeline_error": pipeline_error,
        }
        record.update(verifier_signals)
        return record

    def _record_from_error(
        self,
        sample: BenchmarkSample,
        benchmark: str,
        exc: BaseException,
    ) -> SampleResult:
        """Build a SampleResult for a pipeline crash (fail_on_error=False path)."""
        # Sentinel tier -1 distinguishes a crash from a legitimate Tier 3 RAG
        # route; routing_distribution() excludes tier<1 from the per-tier
        # denominator so Ch5 tier fractions are not inflated by crashes.
        em, f1 = self._score("", sample["answers"], sample.get("gold_label"), benchmark)
        record: SampleResult = {
            "id": sample.get("id", ""),
            "benchmark": benchmark,
            "question": sample["question"],
            "prediction": "",
            "display_answer": "",
            "gold_answers": sample["answers"],
            "gold_label": sample.get("gold_label"),
            "em": em,
            "f1": f1,
            "tier": -1,
            "stored": False,
            "u_stored": None,
            "latency_ms": 0.0,
            "escalated": False,
            "pipeline_error": f"{type(exc).__name__}: {exc}",
        }
        record.update(self._extract_verifier_signals(None))
        return record

    def _run_batched(
        self,
        samples: List[BenchmarkSample],
        benchmark: str,
        store_to_memory: bool,
        cycle: int,
    ) -> List[SampleResult]:
        """Goal 5 Level B batched eval loop.

        Chunks ``samples`` into ``self.batch_size``-sized groups, calls
        ``BatchPipeline.answer_batch`` per chunk, and converts returned
        ``PipelineResult``s into ``SampleResult``s preserving input order.
        Per-chunk exceptions fall back to serial ``_run_one`` for that
        chunk so one pathological batch never aborts the whole eval
        (equivalent to the serial path's per-sample exception handling).
        Memory-store commits still occur in submission order (the
        ``BatchPipeline`` contract), so memory state is deterministic.
        """
        from caem.pipeline_batch import BatchSample

        bp = self._get_batch_pipeline()
        bs = self.batch_size
        results: List[SampleResult] = []
        next_log = 0

        for chunk_start in range(0, len(samples), bs):
            chunk = samples[chunk_start : chunk_start + bs]
            if self.log_every > 0 and chunk_start >= next_log:
                logger.info(
                    "  [%s/%s] %s cycle=%d batched (bs=%d) ...",
                    chunk_start, len(samples), benchmark, cycle, bs,
                )
                next_log = chunk_start + self.log_every

            batch_inputs = [
                BatchSample(
                    query=s["question"],
                    source_benchmark=benchmark,
                    store_to_memory=store_to_memory,
                )
                for s in chunk
            ]

            try:
                batch_results = bp.answer_batch(batch_inputs)
            except Exception as exc:
                if self.fail_on_error:
                    raise
                logger.warning(
                    "BatchPipeline.answer_batch failed for chunk [%d:%d] of %s: %s -- "
                    "falling back to serial for this chunk.",
                    chunk_start, chunk_start + len(chunk), benchmark, exc,
                )
                for s in chunk:
                    results.append(self._run_one(s, benchmark, store_to_memory))
                continue

            if len(batch_results) != len(chunk):
                logger.warning(
                    "BatchPipeline.answer_batch returned %d results for %d inputs "
                    "(chunk [%d:%d] of %s); falling back to serial for this chunk.",
                    len(batch_results), len(chunk),
                    chunk_start, chunk_start + len(chunk), benchmark,
                )
                for s in chunk:
                    results.append(self._run_one(s, benchmark, store_to_memory))
                continue

            for s, pipeline_result in zip(chunk, batch_results):
                results.append(
                    self._record_from_result(s, benchmark, pipeline_result)
                )

        return results

    def _score(
        self,
        prediction: str,
        gold_answers: List[str],
        gold_label: Optional[str],
        benchmark: str,
    ):
        """Return (em, f1) for a prediction given the benchmark type.

        FEVER        -- label extraction + accuracy (f1 == em for 3-class labels)
        TruthfulQA   -- any-match EM via ROUGE-L threshold + best ROUGE-L F1
        StrategyQA   -- boolean yes/no label extraction + EM
        ARC-Challenge -- multiple-choice letter extraction + EM
        TriviaQA     -- any-match EM + best token F1 across all answer aliases
        Natural Questions -- same as TriviaQA (multiple valid answer strings)
        Generic fallback -- single gold answer EM + token F1
        """
        prediction = extract_cot_answer(prediction)

        if benchmark == "fever":
            pred_label = extract_fever_label(prediction)
            gold = gold_label or (gold_answers[0] if gold_answers else "not enough info")
            acc = fever_accuracy(pred_label, gold)
            return acc, acc   # f1 == em for label classification

        elif benchmark == "truthfulqa":
            # EM equivalent uses ROUGE-L threshold > 0.15 since TruthfulQA has no exact string matches.
            f1 = rouge_l(prediction, gold_answers)
            em = float(f1 > 0.15)
            return em, f1

        elif benchmark == "strategyqa":
            # Boolean QA: robust yes/no label extraction from free-form output.
            gold = gold_answers[0] if gold_answers else "no"
            pred_label = extract_strategyqa_label(prediction)
            em = exact_match(pred_label, gold)
            f1 = em   # F1 == EM for binary labels
            return em, f1

        elif benchmark == "arc_challenge":
            gold = gold_answers[0] if gold_answers else ""
            pred_label = extract_arc_label(prediction)
            em = exact_match(pred_label, gold)
            return em, em

        elif benchmark in ("triviaqa", "natural_questions"):
            # Both benchmarks provide multiple valid answer strings (aliases).
            # any_match_em scores 1.0 if prediction matches ANY alias;
            # best_token_f1 returns the highest F1 across all aliases.
            # IMPORTANT: do NOT fall through to the else branch here -- using
            # gold_answers[0] alone silently ignores 10-40 valid aliases per
            # TriviaQA question, producing severely under-counted EM scores.
            em = any_match_em(prediction, gold_answers)
            f1 = best_token_f1(prediction, gold_answers)
            return em, f1

        else:
            # Generic open-ended QA fallback (single gold answer).
            gold = gold_answers[0] if gold_answers else ""
            em = exact_match(prediction, gold)
            f1 = token_f1(prediction, gold)
            return em, f1

    # ------------------------------------------------------------------ #
    # Multi-benchmark convenience runner                                   #
    # ------------------------------------------------------------------ #

    def run_all(
        self,
        samples_by_benchmark: Dict[str, List[BenchmarkSample]],
        cycle: int = 0,
        store_to_memory: bool = False,
    ) -> Dict[str, EvalResult]:
        """Run all benchmarks and return a dict of results.

        Parameters
        ----------
        samples_by_benchmark : dict -- {"fever": [...], "triviaqa": [...], ...}
        cycle : int

        Returns
        -------
        dict -- {"fever": EvalResult, "triviaqa": EvalResult, ...}
        """
        results = {}
        for benchmark, samples in samples_by_benchmark.items():
            results[benchmark] = self.run(benchmark, samples, cycle=cycle, store_to_memory=store_to_memory)
        return results

    # ------------------------------------------------------------------ #
    # Result loading (for post-hoc analysis)                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def load_result(cls, path: str | Path) -> Dict:
        """Load a previously saved JSON result file.

        Parameters
        ----------
        path : str or Path -- path to the .json file saved by run()

        Returns
        -------
        dict with keys "meta" (EvalResult) and "samples" (list of SampleResult)
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def load_all_cycles(
        cls, output_dir: str | Path, benchmark: str
    ) -> List[Dict]:
        """Load all cycle results for one benchmark from an output directory.

        Returns them sorted by cycle number (numeric order).

        Lexical ``sorted(...glob())`` would put ``cycle10.json`` BEFORE
        ``cycle2.json`` (since ASCII ``'1' < '2'``) — latent today because
        Task #80 runs cycles 0-9 only, but immediately incorrect for any
        11+ cycle extension. Downstream BWT / FWT / cycle-over-cycle plots
        would silently receive scrambled data. Audit MAJOR-H1 / Task #116.
        """
        output_dir = Path(output_dir)
        pattern = f"{benchmark}_cycle*.json"

        def _cycle_index(path: Path) -> int:
            # Filename shape: "{benchmark}_cycle{N}.json"; stem removes
            # ".json", split("_cycle") yields ("{benchmark}", "{N}").
            # Defensive: fall back to -1 on malformed names so glob noise
            # doesn't crash the sort.
            try:
                return int(path.stem.split("_cycle")[-1])
            except (ValueError, IndexError):
                return -1

        files = sorted(output_dir.glob(pattern), key=_cycle_index)
        return [cls.load_result(f) for f in files]

    # ------------------------------------------------------------------ #
    # Smoke test (quick sanity check without downloading datasets)        #
    # ------------------------------------------------------------------ #

    def smoke_test(
        self,
        benchmark: str = "fever",
        n: int = 5,
        cycle: int = 0,
    ) -> EvalResult:
        """Run a tiny synthetic eval to verify the harness is wired correctly.

        Parameters
        ----------
        benchmark : str
            Benchmark to use for synthetic samples. Defaults to "fever"
            (primary training benchmark). Use "triviaqa" or "natural_questions"
            to exercise the any_match_em scoring path.
        n : int -- number of synthetic samples
        cycle : int

        Returns
        -------
        EvalResult
        """
        samples = make_synthetic_samples(benchmark=benchmark, n=n)
        return self.run(benchmark=benchmark, samples=samples, cycle=cycle)
