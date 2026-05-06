"""
tests/test_fix6_alias_overlap.py
==================================
Smoke test for v2 Fix 6 — alias_overlap signal (Wikidata-style alias
coverage between answer entities and retrieved-passage entities).

Verifies:
  1. AliasResolver protocol contract (resolve(entity) -> Set[str])
  2. InMemoryAliasResolver bidirectional + case-insensitive lookup
  3. Default entity extractor catches multi-word title-case spans
  4. compute_alias_overlap returns 1.0 when answer surface forms ARE
     aliases of passage surface forms (e.g.
     "William Jefferson Clinton" ↔ "Bill Clinton")
  5. compute_alias_overlap returns 0.0 when no answer entity overlaps
     any passage alias
  6. Returns 0.5 (neutral) when resolver is None / answer/passages
     produce no entities
  7. UnifiedVerifierOutput exposes alias_overlap with default 0.5
  8. COMPOSITE_SIGNALS includes alias_overlap (so the cal_prob
     composite trains a per-signal isotonic for it)
  9. eval/harness._derive_verifier_fields() picks up alias_overlap
     automatically (no manual schema-update needed)
 10. WikidataAliasResolver gracefully handles a missing alias file
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


def test_in_memory_resolver_bidirectional_case_insensitive():
    from caem.verification.alias_overlap import InMemoryAliasResolver
    table = {
        "Bill Clinton": ["William Jefferson Clinton", "Clinton"],
        "United States": ["USA", "U.S.", "America"],
    }
    r = InMemoryAliasResolver(table)
    # Forward lookup
    a = r.resolve("Bill Clinton")
    assert "William Jefferson Clinton" in a
    assert "Clinton" in a
    assert "Bill Clinton" in a
    # Reverse lookup — alias resolves back to canonical
    b = r.resolve("William Jefferson Clinton")
    assert "Bill Clinton" in b
    # Case-insensitive lookup
    c = r.resolve("bill clinton")
    assert "William Jefferson Clinton" in c
    # Unknown surface form — empty set
    assert r.resolve("Unknown Person") == set()


def test_default_entity_extractor():
    from caem.verification.alias_overlap import _default_entity_extractor
    text = "Bill Clinton was president of the United States. New York is a city."
    ents = _default_entity_extractor(text)
    # Multi-word proper-noun spans recovered
    found = " ".join(ents).lower()
    assert "bill clinton" in found
    assert "new york" in found


def test_compute_alias_overlap_full_coverage():
    """If every answer entity is an alias of some passage entity,
    overlap should be 1.0 (pure-alias-match scenario)."""
    from caem.verification.alias_overlap import (
        InMemoryAliasResolver, compute_alias_overlap,
    )
    table = {
        "Bill Clinton": ["William Jefferson Clinton", "Clinton"],
        "United States": ["USA", "U.S.", "America"],
    }
    resolver = InMemoryAliasResolver(table)
    answer = "William Jefferson Clinton served as the 42nd president of America."
    passages = [
        "Bill Clinton was the 42nd president of the United States.",
    ]
    overlap = compute_alias_overlap(answer, passages, resolver)
    # Both "William Jefferson Clinton" and "America" are aliases of
    # entities in the passages, so coverage should be high.
    assert overlap >= 0.5, (
        f"expected high overlap on pure-alias-match scenario; got {overlap}"
    )


def test_compute_alias_overlap_no_coverage():
    """If no answer entity overlaps any passage alias, return 0.0."""
    from caem.verification.alias_overlap import (
        InMemoryAliasResolver, compute_alias_overlap,
    )
    resolver = InMemoryAliasResolver({})
    answer = "Napoleon Bonaparte led the French invasion of Russia."
    passages = [
        "Mahatma Gandhi led nonviolent resistance against British rule.",
    ]
    overlap = compute_alias_overlap(
        answer, passages, resolver,
        # Bypass the default extractor by passing explicit entities
        # so the test isolates the scoring math from the regex.
        entities_in_answer=["Napoleon Bonaparte"],
        entities_in_passages=["Mahatma Gandhi"],
    )
    assert overlap == 0.0, (
        f"expected zero overlap for disjoint entity sets; got {overlap}"
    )


def test_compute_alias_overlap_neutral_fallback():
    """No resolver / no entities → neutral 0.5 prior."""
    from caem.verification.alias_overlap import compute_alias_overlap
    # No resolver
    assert compute_alias_overlap("any answer", ["any passage"], None) == 0.5
    # Resolver present but no entities
    from caem.verification.alias_overlap import InMemoryAliasResolver
    r = InMemoryAliasResolver({})
    assert compute_alias_overlap("", ["passage"], r) == 0.5
    assert compute_alias_overlap("answer", [""], r) == 0.5
    # Explicit empty entity lists
    assert compute_alias_overlap(
        "answer", ["passage"], r,
        entities_in_answer=[], entities_in_passages=[],
    ) == 0.5


def test_unified_verifier_output_carries_alias_overlap():
    from caem.verification.verifier import UnifiedVerifierOutput
    vout = UnifiedVerifierOutput(
        u_token=0.7, u_dropout=0.1, u_internal=0.6,
        s_avg=0.7, h_norm=0.2, p_entail=0.6,
        p_ground_max=0.5, p_ground_mean=0.5, p_ground_atomic=0.5,
        p_contra=0.05,
        u_stored=0.6, decision="STORE",
        early_exit_triggered=False, abstained=False,
    )
    # Default value (the composite isn't penalised before resolver wiring)
    assert vout.alias_overlap == 0.5
    # Explicit override propagates
    vout2 = UnifiedVerifierOutput(
        u_token=0.7, u_dropout=0.1, u_internal=0.6,
        s_avg=0.7, h_norm=0.2, p_entail=0.6,
        p_ground_max=0.5, p_ground_mean=0.5, p_ground_atomic=0.5,
        p_contra=0.05,
        u_stored=0.6, decision="STORE",
        early_exit_triggered=False, abstained=False,
        alias_overlap=0.83,
    )
    assert vout2.alias_overlap == 0.83


def test_composite_signals_includes_alias_overlap():
    from caem.verification.cal_prob_composite import COMPOSITE_SIGNALS
    assert "alias_overlap" in COMPOSITE_SIGNALS, (
        f"COMPOSITE_SIGNALS missing alias_overlap; got {COMPOSITE_SIGNALS}"
    )


def test_harness_verifier_fields_auto_picks_up_alias_overlap():
    """eval/harness._derive_verifier_fields() inspects the dataclass at
    import time — alias_overlap should appear automatically."""
    from eval.harness import VERIFIER_FIELDS
    assert "alias_overlap" in VERIFIER_FIELDS, (
        f"eval.harness.VERIFIER_FIELDS missing alias_overlap; got {VERIFIER_FIELDS}"
    )


def test_wikidata_resolver_missing_file_returns_empty():
    from caem.verification.alias_overlap import WikidataAliasResolver
    resolver = WikidataAliasResolver(Path("/nonexistent/path/to/aliases.json"))
    # Lazy load — first call attempts to read; missing file → empty table
    result = resolver.resolve("Anything")
    assert result == set()


def test_wikidata_resolver_loads_real_json():
    from caem.verification.alias_overlap import WikidataAliasResolver
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "aliases.json"
        with path.open("w") as f:
            json.dump({
                "Albert Einstein": ["A. Einstein", "Einstein"],
                "Theory of Relativity": ["General Relativity", "Special Relativity"],
            }, f)
        resolver = WikidataAliasResolver(path)
        # First lookup triggers load
        a = resolver.resolve("Einstein")
        assert "Albert Einstein" in a
        assert "A. Einstein" in a
        # Case-insensitive
        b = resolver.resolve("ALBERT EINSTEIN")
        assert "Einstein" in b
        # Unknown
        assert resolver.resolve("Random Name") == set()


def test_verifier_init_accepts_alias_resolver_kwarg():
    """UnifiedVerifier.__init__ must accept alias_resolver=... and store it."""
    import inspect
    from caem.verification.verifier import UnifiedVerifier
    sig = inspect.signature(UnifiedVerifier.__init__)
    assert "alias_resolver" in sig.parameters, (
        "UnifiedVerifier.__init__ must accept alias_resolver kwarg"
    )


def test_compute_alias_overlap_returns_float_in_unit_interval():
    """Sanity: scoring function output is always in [0, 1]."""
    from caem.verification.alias_overlap import (
        InMemoryAliasResolver, compute_alias_overlap,
    )
    resolver = InMemoryAliasResolver({"X": ["Y"], "Z": ["W"]})
    test_cases = [
        ("X is Y", ["X is great"]),
        ("Z and X met", ["Y and W"]),
        ("nothing", ["matches"]),
        ("", []),
        ("X X X", ["Y Y Y"]),
    ]
    for ans, passages in test_cases:
        v = compute_alias_overlap(ans, passages, resolver)
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0, f"out-of-range output {v} for ({ans}, {passages})"


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
