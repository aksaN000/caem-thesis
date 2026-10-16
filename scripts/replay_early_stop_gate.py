"""Replay the cycle-five early-stop gate under the realised and the corrected wiring.

Read-only. Inputs: outputs/full_run/experiment_summary.csv (per-cycle EM and MMLU
accuracy) and outputs/tab_decision_matrix_trajectory.csv (per-cycle stream-pass
STORE decisions). Output: outputs/analysis/early_stop_gate_replay.json.

The realised run (outputs/full_run/experiment.log, "equilibrium gate | cycle=5")
fed the gate the evaluation-pass storage rate (always 0, the evaluation pass never
stores) and raw MMLU accuracy against the 0.93 ratio floor. Both wiring faults were
fixed in scripts/run_experiment.py on 2026-10-16.
"""
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caem.eval.equilibrium import three_signal_gate  # noqa: E402

em = collections.defaultdict(list)
mmlu = {}
for r in csv.DictReader(open(ROOT / "outputs/full_run/experiment_summary.csv")):
    c = int(r["cycle"])
    em[c].append(float(r["em"]))
    if r["mmlu_retention_pct"]:
        mmlu[c] = float(r["mmlu_retention_pct"]) / 100.0
store = collections.defaultdict(lambda: [0, 0])
for r in csv.DictReader(open(ROOT / "outputs/tab_decision_matrix_trajectory.csv")):
    c = int(r["cycle"])
    s = int(r["STORE_correct"]) + int(r["STORE_wrong"])
    n = s + sum(int(r[k]) for k in ("DEFERRED_correct", "DEFERRED_wrong", "DISCARD_correct", "DISCARD_wrong"))
    store[c][0] += s
    store[c][1] += n

cycles = sorted(em)
ces = [sum(em[c]) / len(em[c]) for c in cycles]
mmlu_acc = [mmlu[c] for c in cycles]
ratios = [m / mmlu_acc[0] for m in mmlu_acc]
stream_rate = [store[c][0] / store[c][1] for c in cycles]

realised = three_signal_gate(ces, [0.0] * len(cycles), mmlu_acc, cycle=cycles[-1])
fixed = three_signal_gate(ces, stream_rate, ratios, cycle=cycles[-1])
out = {
    "cycles": cycles,
    "ces_mean_em": [round(x, 4) for x in ces],
    "mmlu_accuracy": mmlu_acc,
    "mmlu_retention_ratio": [round(x, 4) for x in ratios],
    "stream_store_rate": [round(x, 4) for x in stream_rate],
    "realised_wiring": {"signals_fired": realised.signals_fired, "should_stop": realised.should_stop},
    "corrected_wiring": {"signals_fired": fixed.signals_fired, "should_stop": fixed.should_stop},
}
(ROOT / "outputs/analysis").mkdir(parents=True, exist_ok=True)
json.dump(out, open(ROOT / "outputs/analysis/early_stop_gate_replay.json", "w"), indent=2)
print(json.dumps(out, indent=2))
