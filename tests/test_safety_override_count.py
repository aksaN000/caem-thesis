"""Unit test for the safety-override counting in scripts/aggregate_section_5_4.py.

The aggregator builds its tables at import time (it needs outputs/), so the two
pure helpers are extracted from its source with ``ast`` and run on synthetic
samples. Runs under pytest or directly: ``python tests/test_safety_override_count.py``.
"""
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "aggregate_section_5_4.py"
_WANTED = {"safety_floors_by_cycle", "count_safety_fires"}


def _load_helpers():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _WANTED]
    assert {f.name for f in funcs} == _WANTED, "helper functions missing from aggregate_section_5_4.py"
    ns = {"json": json, "Path": Path}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(SCRIPT), "exec"), ns)
    return ns["safety_floors_by_cycle"], ns["count_safety_fires"]


def test_count_safety_fires_strict_inequality_and_missing_u_pre():
    _, count = _load_helpers()
    samples = [{"u_pre": 0.30}, {"u_pre": 0.38}, {"u_pre": 0.379}, {"u_pre": None},
               {}, {"u_pre": 0.9, "escalated": True}]
    fires, n, min_u = count(samples, 0.38)
    assert fires == 2          # 0.30 and 0.379; 0.38 is not < 0.38; escalated ignored
    assert n == 6              # samples without u_pre still count toward n
    assert min_u == 0.30
    assert count([], 0.38) == (0, 0, None)


def test_floors_carry_forward_over_aborted_cycle_and_fallback_to_pooled():
    floors_fn, count = _load_helpers()
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "calibrated_config.json").write_text(json.dumps({"safety_u_pre_min_per_benchmark": {}}))
        (d / "calibrated_config_cycle3.json").write_text(
            json.dumps({"safety_u_pre_min_per_benchmark": {"commonsense_qa": 0.491}}))
        # cycle 4 aborted: no file -> keeps cycle 3's floors
        (d / "calibrated_config_cycle5.json").write_text(
            json.dumps({"safety_u_pre_min_per_benchmark": {"commonsense_qa": 0.5687}}))
        benches = ["fever", "commonsense_qa", "strategyqa"]
        fl = floors_fn(d, range(6), benches, 0.38)
    assert [fl[c]["commonsense_qa"] for c in range(6)] == [0.38, 0.38, 0.38, 0.491, 0.491, 0.5687]
    assert all(fl[c]["fever"] == 0.38 and fl[c]["strategyqa"] == 0.38 for c in range(6))

    # Synthetic replay of the main-run shape: CSQA fires only once its floor rises.
    csqa = {c: [{"u_pre": 0.45}] * {3: 12, 4: 12}.get(c, 0) + [{"u_pre": 0.55}] * {5: 38}.get(c, 0)
            + [{"u_pre": 0.80}] * 250 for c in range(6)}
    per_cycle = [count(csqa[c], fl[c]["commonsense_qa"])[0] for c in range(6)]
    assert per_cycle == [0, 0, 0, 12, 12, 38]
    assert sum(per_cycle) == 62


def test_missing_calibration_dir_gives_pooled_everywhere():
    floors_fn, _ = _load_helpers()
    fl = floors_fn(Path("/nonexistent/calibration"), range(3), ["commonsense_qa"], 0.38)
    assert fl == {0: {"commonsense_qa": 0.38}, 1: {"commonsense_qa": 0.38}, 2: {"commonsense_qa": 0.38}}


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
