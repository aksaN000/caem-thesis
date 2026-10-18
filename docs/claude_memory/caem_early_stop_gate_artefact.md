---
name: caem_early_stop_gate_artefact
description: "C5 early stop fired on two mis-wired signals (NOT evidence of saturation). 2026-10-17 author REVISED: remove the disclosure from the thesis; say only 'stopped after C5 as planned'; never present the gate as saturation evidence."
metadata:
  node_type: memory
  type: project
  originSessionId: 9bbed419-78c6-48fb-ad84-30c1f48abc6e
  modified: 2026-10-16T07:39:55.727Z
---

Main run log `outputs/full_run/experiment.log` L70711: `equilibrium gate | cycle=5 | signals=storage_saturation,mmlu_floor | stop=True` (2-of-3 rule, `caem/eval/equilibrium.py::three_signal_gate`, called from `scripts/run_experiment.py` ~L2046–2100; burn-in `early_stop_min_cycles=5`).

**Both fired signals are artefacts:**
- storage_saturation: fed the mean of per-benchmark `storage_rate` from the eval harness, which never writes memory → 0.00% every cycle (< 0.05 always). The real stream-chunk admission rate was ~50–53% (appendix C tab:app-per-cycle-memory).
- mmlu_floor: fed `mmlu_per_cycle` = raw MMLU accuracy (0.635, 0.595, 0.605, 0.621, 0.609) compared with `forgetting_tolerance` 0.93, a ratio floor → always true. Real MMLU retention ratios: 1.00, 0.937, 0.953, 0.978, 0.959 (experiment_summary.csv), never ≤ 0.93.
- So the gate was certain to fire at the first cycle it ran (C5), whatever the trajectory did. The third signal (CES/EM gradient) did not fire.
- Consistent with [[caem_cycle4_retention_abort]]: the author had already decided (2026-05-14) to stop at C5 regardless.

**Status:** APPROVED by Aksan 2026-10-16 ("thesis should be as transparent as it is supposed to be. we can mislead other fellow researchers even if our contribution becomes the cost"). Disclose: run stopped after C5 as planned; gate fired at the same point but is not evidence of saturation. Done in ch3 (sec:env-cycles, risk table); TODO ch4, ch5 (L104, L609), ch6, appendix C (L94 'STOP verdict').

**Code fixed 2026-10-16** (review/ch3, commit cd2d4da): `scripts/run_experiment.py` now feeds the Step 4 stream-pass admission rate (sample-weighted) and the MMLU ratio current/pristine. Replay `scripts/replay_early_stop_gate.py` → `outputs/analysis/early_stop_gate_replay.json` (force-added; outputs/ is git-ignored): realised wiring fires storage_saturation + mmlu_floor; corrected wiring fires nothing at C5. Thesis must say "corrected after the run" and cite the replay.

**REVISED 2026-10-17 (author):** remove the mis-wiring disclosure from the thesis ("doesn't add any value but shows our carelessness. we would've stopped at cycle 5 anyways"). Done in ch1/3/4/5 + appendix C (outcome column instead of verdicts). Ch6/abstract: say only 'stopped after cycle five as planned'; never cite the gate verdict as saturation. Fault kept in repo tracker C1.
