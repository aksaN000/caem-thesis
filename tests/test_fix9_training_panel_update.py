"""
tests/test_fix9_training_panel_update.py
==========================================
Smoke test for v2 Fix 9: training panel update + new benchmark loaders.

Verifies:
  1. TRAINING_BENCHMARKS = (fever, triviaqa, hotpotqa, commonsense_qa)
  2. TRANSFER_BENCHMARKS = (truthfulqa, strategyqa, natural_questions)
  3. caem.benchmark_splits.TRAINING_BENCHMARKS imports from caem.config
  4. _EVAL_SPLIT_MAP has entries for hotpotqa + commonsense_qa
  5. PER_BENCHMARK_TRAIN_CHUNK_SIZE: CSQA at 700, others at 1000
  6. eval.benchmarks exposes load_hotpotqa + load_commonsense_qa
  7. load_benchmark dispatcher includes new benchmarks
  8. make_synthetic_samples produces samples for new benchmarks
  9. scripts/seed_cold_start.py imports TRAINING_BENCHMARKS dynamically
 10. run_phase1a.sh BENCHMARKS array contains v2 panel
"""
from __future__ import annotations

import sys


def test_training_benchmarks_v2_panel():
    """TRAINING_BENCHMARKS == v2 four-task panel."""
    from caem.config import TRAINING_BENCHMARKS
    assert TRAINING_BENCHMARKS == ("fever", "triviaqa", "hotpotqa", "commonsense_qa"), (
        f"v2 TRAINING_BENCHMARKS expected (fever, triviaqa, hotpotqa, commonsense_qa); "
        f"got {TRAINING_BENCHMARKS}"
    )


def test_transfer_benchmarks_v2_panel():
    """TRANSFER_BENCHMARKS == v2 three-benchmark transfer panel."""
    from caem.config import TRANSFER_BENCHMARKS
    assert TRANSFER_BENCHMARKS == ("truthfulqa", "strategyqa", "natural_questions"), (
        f"v2 TRANSFER_BENCHMARKS expected (truthfulqa, strategyqa, natural_questions); "
        f"got {TRANSFER_BENCHMARKS}"
    )


def test_benchmark_splits_imports_from_config():
    """caem.benchmark_splits.TRAINING_BENCHMARKS must equal caem.config.TRAINING_BENCHMARKS
    (single source of truth — was duplicated in v1, drifted across the trajectory)."""
    from caem.config import TRAINING_BENCHMARKS as cfg_train
    from caem.config import TRANSFER_BENCHMARKS as cfg_transfer
    from caem.benchmark_splits import TRAINING_BENCHMARKS as splits_train
    from caem.benchmark_splits import TRANSFER_BENCHMARKS as splits_transfer
    assert splits_train is cfg_train, (
        "benchmark_splits.TRAINING_BENCHMARKS must BE config.TRAINING_BENCHMARKS "
        "(import equality, not just value equality) to prevent drift."
    )
    assert splits_transfer is cfg_transfer


def test_eval_split_map_has_new_benchmarks():
    """_EVAL_SPLIT_MAP has entries for hotpotqa + commonsense_qa."""
    from caem.benchmark_splits import _EVAL_SPLIT_MAP
    assert "hotpotqa" in _EVAL_SPLIT_MAP, "hotpotqa missing from _EVAL_SPLIT_MAP"
    assert "commonsense_qa" in _EVAL_SPLIT_MAP, "commonsense_qa missing from _EVAL_SPLIT_MAP"
    assert _EVAL_SPLIT_MAP["hotpotqa"] in ("validation", "dev"), (
        f"hotpotqa eval split should be validation or dev; got {_EVAL_SPLIT_MAP['hotpotqa']}"
    )
    assert _EVAL_SPLIT_MAP["commonsense_qa"] in ("validation", "dev"), (
        f"commonsense_qa eval split should be validation or dev; got {_EVAL_SPLIT_MAP['commonsense_qa']}"
    )


def test_per_benchmark_train_chunk_override():
    """PER_BENCHMARK_TRAIN_CHUNK_SIZE: CSQA at 700, others at 1000."""
    from caem.benchmark_splits import PER_BENCHMARK_TRAIN_CHUNK_SIZE, DEFAULT_TRAIN_CHUNK_SIZE
    assert PER_BENCHMARK_TRAIN_CHUNK_SIZE["fever"] == 1000
    assert PER_BENCHMARK_TRAIN_CHUNK_SIZE["triviaqa"] == 1000
    assert PER_BENCHMARK_TRAIN_CHUNK_SIZE["hotpotqa"] == 1000
    assert PER_BENCHMARK_TRAIN_CHUNK_SIZE["commonsense_qa"] == 700, (
        f"CSQA stream chunk should be 700 (small training pool); "
        f"got {PER_BENCHMARK_TRAIN_CHUNK_SIZE['commonsense_qa']}"
    )
    assert DEFAULT_TRAIN_CHUNK_SIZE == 1000, (
        f"v2 default should be 1000 (was 5000 in v1); got {DEFAULT_TRAIN_CHUNK_SIZE}"
    )


def test_eval_benchmarks_exposes_new_loaders():
    """eval.benchmarks has load_hotpotqa and load_commonsense_qa."""
    import eval.benchmarks as evb
    assert hasattr(evb, "load_hotpotqa"), "load_hotpotqa missing from eval.benchmarks"
    assert hasattr(evb, "load_commonsense_qa"), "load_commonsense_qa missing from eval.benchmarks"
    assert callable(evb.load_hotpotqa)
    assert callable(evb.load_commonsense_qa)


def test_load_benchmark_dispatcher_includes_new():
    """load_benchmark dispatches by name for hotpotqa + commonsense_qa."""
    import eval.benchmarks as evb
    import inspect
    src = inspect.getsource(evb.load_benchmark)
    assert "hotpotqa" in src, "load_benchmark dispatcher missing hotpotqa branch"
    assert "commonsense_qa" in src, "load_benchmark dispatcher missing commonsense_qa branch"


def test_make_synthetic_samples_new_benchmarks():
    """Synthetic factory produces correct schema for hotpotqa + commonsense_qa."""
    from eval.benchmarks import make_synthetic_samples

    hotpot = make_synthetic_samples("hotpotqa", n=3, seed=0)
    assert len(hotpot) == 3, "hotpotqa synthetic should produce n samples"
    for s in hotpot:
        assert s["benchmark"] == "hotpotqa"
        assert "question" in s
        assert isinstance(s["answers"], list)
        assert len(s["answers"]) >= 1

    csqa = make_synthetic_samples("commonsense_qa", n=5, seed=0)
    assert len(csqa) == 5, "commonsense_qa synthetic should produce n samples"
    labels_seen = set()
    for s in csqa:
        assert s["benchmark"] == "commonsense_qa"
        assert s["gold_label"] in ("A", "B", "C", "D", "E"), (
            f"CSQA must use 5-choice labels; got {s['gold_label']}"
        )
        labels_seen.add(s["gold_label"])
        # Verify the prompt enumerates all 5 options A-E
        for letter in ("A", "B", "C", "D", "E"):
            assert f"({letter})" in s["question"], (
                f"CSQA prompt should enumerate option ({letter}); got: {s['question']}"
            )
    assert labels_seen >= {"A", "B"}, "CSQA synthetic should cycle through labels"


def test_seed_cold_start_imports_dynamic_training_benchmarks():
    """seed_cold_start.py uses dynamic TRAINING_BENCHMARKS, not hardcoded names."""
    import importlib.util
    spec = importlib.util.find_spec("scripts.seed_cold_start")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        src = f.read()
    # Must NOT have the v1 hardcoded check
    assert 'benchmark in ("fever", "triviaqa", "natural_questions")' not in src, (
        "seed_cold_start.py still has the v1 hardcoded benchmark tuple check; "
        "Fix 9 was supposed to make this dynamic via TRAINING_BENCHMARKS."
    )
    # Must use TRAINING_BENCHMARKS
    assert "in TRAINING_BENCHMARKS" in src, (
        "seed_cold_start.py should use 'in TRAINING_BENCHMARKS' for the dynamic check."
    )


def test_run_phase1a_benchmarks_array_v2():
    """run_phase1a.sh BENCHMARKS array contains v2 panel."""
    with open("/workspace/caem/run_phase1a.sh") as f:
        src = f.read()
    # v2 panel: 4 training + 3 transfer = 7 benchmarks total
    assert "BENCHMARKS=(fever triviaqa hotpotqa commonsense_qa truthfulqa strategyqa natural_questions)" in src, (
        "run_phase1a.sh BENCHMARKS array does not match v2 panel."
    )
    # The v1 panel must be gone
    assert "BENCHMARKS=(fever triviaqa natural_questions truthfulqa strategyqa arc_challenge asqa)" not in src, (
        "run_phase1a.sh still contains the v1 BENCHMARKS array; Fix 9 incomplete."
    )


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
