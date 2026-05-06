"""
tests/test_fix2_verifier_api.py
=================================
Smoke test for v2 Fix 2 (Commit A) — verifier API change.

This commit adds the source_benchmark parameter to UnifiedVerifier.verify(),
verify_batch(), and _composite(). All callsites are updated to thread the
benchmark tag through. CalProbComposite.predict() may or may not accept
the kwarg yet — Fix 2 falls back to the legacy signature on TypeError so
this commit is functionally equivalent to v1 (predict ignores
source_benchmark, single global calibration). Commit B will land the
per-benchmark CalProbComposite to actually use the parameter.

Verifies:
  1. UnifiedVerifier.verify() signature accepts source_benchmark kwarg
  2. UnifiedVerifier.verify_batch() signature accepts source_benchmarks list
  3. UnifiedVerifier._composite() signature accepts source_benchmark kwarg
  4. CAEMPipeline._verify() signature accepts source_benchmark kwarg
  5. pipeline.answer() threads source_benchmark to _verify (source pattern)
  6. make_retroverify_fn closure passes entry.source_benchmark to verify()
  7. make_reconsider_deferred_fn closure passes
     deferred_entry.source_benchmark to verify()
  8. _composite() forwards source_benchmark to CalProbComposite.predict()
  9. The legacy CalProbComposite.predict(signal_values) signature still
     works (TypeError fallback in _composite())
"""
from __future__ import annotations

import inspect
import sys


def test_verify_signature_has_source_benchmark():
    from caem.verification.verifier import UnifiedVerifier
    sig = inspect.signature(UnifiedVerifier.verify)
    assert "source_benchmark" in sig.parameters, (
        f"UnifiedVerifier.verify() missing source_benchmark; "
        f"params: {list(sig.parameters.keys())}"
    )
    p = sig.parameters["source_benchmark"]
    assert p.default is None, f"source_benchmark default should be None; got {p.default}"


def test_verify_batch_signature_has_source_benchmarks():
    from caem.verification.verifier import UnifiedVerifier
    sig = inspect.signature(UnifiedVerifier.verify_batch)
    assert "source_benchmarks" in sig.parameters, (
        f"UnifiedVerifier.verify_batch() missing source_benchmarks list; "
        f"params: {list(sig.parameters.keys())}"
    )


def test_composite_signature_has_source_benchmark():
    from caem.verification.verifier import UnifiedVerifier
    sig = inspect.signature(UnifiedVerifier._composite)
    assert "source_benchmark" in sig.parameters


def test_pipeline_verify_signature_has_source_benchmark():
    from caem.pipeline import CAEMPipeline
    sig = inspect.signature(CAEMPipeline._verify)
    assert "source_benchmark" in sig.parameters


def test_pipeline_answer_threads_source_benchmark_in_source():
    """pipeline.answer() must pass source_benchmark to _verify at both
    Tier 2 and Tier 3 verify call sites."""
    import importlib.util
    spec = importlib.util.find_spec("caem.pipeline")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        src = f.read()
    # Both Tier 2 and Tier 3 _verify calls must pass source_benchmark.
    n = src.count("source_benchmark=source_benchmark")
    assert n >= 2, (
        f"Expected at least 2 'source_benchmark=source_benchmark' threading "
        f"sites in pipeline.py (Tier 2 + Tier 3 _verify); found {n}"
    )


def test_retroverify_closure_threads_source_benchmark():
    """make_retroverify_fn closure must pass entry.source_benchmark to
    verifier.verify()."""
    import importlib.util
    spec = importlib.util.find_spec("caem.pipeline")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        src = f.read()
    assert 'source_benchmark=getattr(entry, "source_benchmark", None)' in src, (
        "make_retroverify_fn closure does not thread "
        "entry.source_benchmark to verifier.verify()."
    )


def test_reconsider_closure_threads_source_benchmark():
    """make_reconsider_deferred_fn closure must pass
    deferred_entry.source_benchmark to verifier.verify()."""
    import importlib.util
    spec = importlib.util.find_spec("caem.pipeline")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        src = f.read()
    assert 'source_benchmark=getattr(deferred_entry, "source_benchmark", None)' in src, (
        "make_reconsider_deferred_fn closure does not thread "
        "deferred_entry.source_benchmark to verifier.verify()."
    )


def test_composite_forwards_source_benchmark_to_calprob():
    """_composite() must call self._cal_prob_composite.predict() with
    source_benchmark=source_benchmark, with a TypeError fallback to the
    legacy signature for backward-compat."""
    import importlib.util
    spec = importlib.util.find_spec("caem.verification.verifier")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        src = f.read()
    assert "source_benchmark=source_benchmark" in src, (
        "_composite() does not forward source_benchmark to CalProbComposite.predict()"
    )
    assert "except TypeError:" in src, (
        "_composite() missing TypeError fallback for legacy "
        "CalProbComposite.predict() signature"
    )


def test_legacy_calprob_predict_signature_still_works():
    """CalProbComposite.predict(signal_values) without source_benchmark
    kwarg must still work (commit A keeps legacy single-global path)."""
    import inspect
    from caem.verification.cal_prob_composite import CalProbComposite
    sig = inspect.signature(CalProbComposite.predict)
    # Either the legacy signature (no source_benchmark) — TypeError
    # fallback handles it. Or the new signature (with optional
    # source_benchmark) — predict accepts it.
    # Either is acceptable for commit A.
    params = list(sig.parameters.keys())
    assert "signal_values" in params, (
        f"CalProbComposite.predict() must accept signal_values dict; "
        f"params: {params}"
    )


def test_imports_clean_after_api_change():
    """All v2 modules import cleanly after the API change."""
    import caem.verification.verifier
    import caem.verification.cal_prob_composite
    import caem.pipeline
    import scripts.run_experiment
    import caem.training.self_improvement
    import eval.benchmarks


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL  {name}: {e}")
            except Exception as e:
                failures += 1
                print(f"  ERR   {name}: {type(e).__name__}: {e}")
    if failures:
        print(f"\n{failures} test(s) failed.")
        sys.exit(1)
    print(f"\nAll tests passed.")
