---
name: Phase 1d cycle_1 adapter LOST in gdrive (4 files overwritten by Phase 1e)
description: 4 files in v2_1_phase1d_archive/full_run/cycle_1/ were overwritten by Phase 1e cycle_1 outputs on 2026-05-20. Phase 1d eval JSONs + CSVs untouched. Zero thesis impact.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Finding logged 2026-05-29.**

**The contamination:**
On 2026-05-20 03:55:27, the Phase 1e cycle_1 training run wrote its adapter outputs into the **Phase 1d archive** folder (probably because a hardcoded gdrive path in the offload step pointed at the archive). Exactly 4 files were overwritten:

- `gdrive:caem-phase1a/v2_1_phase1d_archive/full_run/cycle_1/meta.pkl`
- `gdrive:caem-phase1a/v2_1_phase1d_archive/full_run/cycle_1/adapter/adapter_config.json`
- `gdrive:caem-phase1a/v2_1_phase1d_archive/full_run/cycle_1/adapter/adapter_model.safetensors`
- `gdrive:caem-phase1a/v2_1_phase1d_archive/full_run/cycle_1/adapter/README.md`

These files now contain **Phase 1e cycle_1 adapter weights**, not Phase 1d's.

**What's recoverable:**
- Phase 1d cycle_0 adapter: ✓ preserved at `gdrive:caem-phase1a/pre_main_snapshot/700b9e3_phase1d/cycle_0/`
- Phase 1d cycle_2..6 adapters: ✓ untouched in archive
- Phase 1d cycle_1 ADAPTER: ✗ LOST from gdrive. Not in HF (HF snapshot is Phase 1a). Would need re-training to recover.

**What's NOT lost (still intact in v2_1_phase1d_archive):**
- All Phase 1d cycle_1..6 eval JSONs (predictions, EM, CHM)
- Phase 1d trajectory CSVs (purity, retention, calibration)
- Phase 1d baselines_v1_panel/ — all 8 baseline results

**Thesis impact: ZERO.**
- Phase 1d is archived as "prior phase that motivated Phase 1e redo", not used for any headline claim.
- Phase 1d cycle_1 ADAPTER weights would only be needed to re-run inference on Phase 1d cycle_1's model — no reason to do this.
- All Phase 1d numbers that DO matter for context (EM, CHM trajectories) live in the eval JSONs + CSVs, which are intact.

**Confirmed contamination scope: exactly 4 files.**
Sorted-by-date scan of `v2_1_phase1d_archive/` shows the next-oldest file is `calibration_trajectory.csv` at 2026-05-17 22:52 — 2 days before Phase 1e cycle_1 ran. So only these 4 files were affected; nothing else slipped in.

**Related gdrive structure observations (also logged 2026-05-29):**
- `gdrive:caem-phase1a/v2_1_phase1d_archive/` — correct Phase 1d archive (modulo the 4 contaminated files)
- `gdrive:caem-phase1a/v2_1_phase1d/` — MISLABELED. Cycle_{1..5} dates 2026-05-21 to 2026-05-25 are all Phase 1e timing. This folder is Phase 1e data uploaded under the wrong name. Should rename to `v2_1_phase1e_main/`.
- `gdrive:caem-phase1a/v2_1_phase1e_2026-05-19/ruc_build/` — separate RUC-build snapshot, untouched.

**Action items:**
1. Don't try to use Phase 1d cycle_1 adapter weights — they're Phase 1e's.
2. Reorganize gdrive: rename `v2_1_phase1d/` → `v2_1_phase1e_main/` once local mirror is verified.
3. Add a README to `v2_1_phase1d_archive/full_run/cycle_1/` noting the 4-file contamination so future-you doesn't load them by accident.
4. The bug that caused the contamination (hardcoded archive path in offload step) should be fixed in any future runner, but is moot now since Phase 1 is done.
