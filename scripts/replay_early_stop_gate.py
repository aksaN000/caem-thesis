"""Replay the cycle-five early-stop gate under the realised and the corrected wiring.

Read-only. Inputs: outputs/full_run/experiment_summary.csv (per-cycle EM and MMLU
accuracy) and outputs/full_run/eval/<bench>_cycle<N>_streamchunk.json (the Step 4
stream pass; ``storage_rate`` there is the fraction actually inserted into memory,
the quantity the corrected runner feeds the gate). Output:
outputs/analysis/early_stop_gate_replay.json.

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
store = collections.defaultdict(lambda: [0.0, 0])
for f in sorted((ROOT / "outputs/full_run/eval").glob("*_cycle*_streamchunk.json")):
    c = int(f.name.split("_cycle")[1].split("_")[0])
    meta = json.load(open(f))["meta"]
    store[c][0] += float(meta["storage_rate"]) * int(meta["n"])
    store[c][1] += int(meta["n"])

cycles = sorted(em)
ces = [sum(em[c]) / len(em[c]) for c in cycles]
mmlu_acc = [mmlu[c] for c in cycles]
ratios = [m / mmlu_acc[0] for m in mmlu_acc]
# Cycle 0 has no stream pass (cold-start seed); the gate only reads the latest value.
stream_rate = [store[c][0] / store[c][1] if store[c][1] else float("nan") for c in cycles]

realised = three_signal_gate(ces, [0.0] * len(cycles), mmlu_acc, cycle=cycles[-1])
fixed = three_signal_gate(ces, stream_rate, ratios, cycle=cycles[-1])
out = {
    "cycles": cycles,
    "ces_mean_em": [round(x, 4) for x in ces],
    "mmlu_accuracy": mmlu_acc,
    "mmlu_retention_ratio": [round(x, 4) for x in ratios],
    "stream_admission_rate": [None if x != x else round(x, 4) for x in stream_rate],
    "realised_wiring": {"signals_fired": realised.signals_fired, "should_stop": realised.should_stop},
    "corrected_wiring": {"signals_fired": fixed.signals_fired, "should_stop": fixed.should_stop},
}
(ROOT / "outputs/analysis").mkdir(parents=True, exist_ok=True)
json.dump(out, open(ROOT / "outputs/analysis/early_stop_gate_replay.json", "w"), indent=2)
print(json.dumps(out, indent=2))
