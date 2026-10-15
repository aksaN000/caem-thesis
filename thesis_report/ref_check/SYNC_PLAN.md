# Making local, GitHub and Drive consistent

Goal: the local repo, GitHub and Google Drive hold the same final code, data and thesis, and every
number in the thesis traces to that data.

## Source of truth

| What | Canonical copy | Why |
|---|---|---|
| Experiment data | **local `outputs/`** | The only copy with the 13 June B6/B7 fix and the post-rerun B8 outputs. Regenerating from it reproduces the thesis tables. The Drive copy gives the pre-fix B6 (0.325) and a zero-shot copy of B8. |
| Code + thesis source | **GitHub `branch/ruc-classifier` (`da05919`)** + the audit branch `claude/elegant-planck-strk55` | `main` is from April 2026 and is not the thesis. |
| Thesis numbers | the committed `figures/auto` tables | Checked against local `outputs/`: no wrong numbers. The open items are the safety-override table, panel (b) of `tab_tier_trajectory`, and `tab_halluc_subtypes` (not yet re-checked). |

**Rule:** data flows local → Drive, and code and text flow through GitHub. Nothing is copied from Drive back into
local `outputs/`. Never regenerate `tab_theorem_receipts_summary` or `sweep_top` from their current scripts:
both scripts still hold stale hard-coded text.

## Steps (local agent: run steps 1 to 3 and report back; do not upload or push in these steps)

1. **Repository state.** Run the following and paste the output:
   `git status`, `git log -1 --oneline`, `git branch -vv`, `git remote -v`.
   If `HEAD` is not `da05919` or the working tree is not clean, say so before anything else.
2. **Local vs Drive data.** Find where the Drive folder `caem-phase1a` is mounted on this machine:
   - Google Drive for desktop, e.g. `/mnt/g/My Drive/caem-phase1a` from WSL;
   - or an rclone mount.

   Then run, from the repo root:
   ```bash
   python thesis_report/ref_check/drive_sync_check.py "<mounted path>/caem-phase1a"
   ```
   Upload `~/caem_drive_sync.md` to Drive. The script is read-only and compares every file by SHA-256. If no
   mount exists, report that instead.
3. **Logs and the remaining table check.**
   - Run `python thesis_report/ref_check/log_value_search.py` and upload `~/caem_log_search.md`.
   - Re-run `python thesis_report/ref_check/local_regen_check.py`; this version also covers `tab_halluc_subtypes`.
     Upload the new `regen_diff.txt` and `state.txt` from `~/caem_regen_check/` as plain files, not as a zip.

   Fetch the scripts with
   `git fetch origin claude/elegant-planck-strk55 && git checkout origin/claude/elegant-planck-strk55 -- thesis_report/ref_check/`.
   This adds only the `ref_check/` folder to your working tree.

## Then (after the user approves each step)

4. **Thesis text fixes** go on `claude/elegant-planck-strk55`, each logged before → after in `fixes_applied.md`.
   Data-independent fixes come first. The fixes that depend on steps 2 and 3 follow once those results are in.
5. **GitHub:** open a PR from `claude/elegant-planck-strk55` into `branch/ruc-classifier`. The user reviews and merges it.
   Optionally make `branch/ruc-classifier` the default branch, or fast-forward `main` to it, so the repo front
   page shows the thesis version.
6. **Local:** `git pull` the merged branch and set its upstream (`git branch -u origin/branch/ruc-classifier`).
   Rebuild `main.pdf` and re-run `local_regen_check.py`. Expect zero genuine differences.
7. **Drive:** upload the files that step 2 lists as DIFFER or LOCAL ONLY from local `outputs/` to the mapped
   Drive folders.
   - Do not delete old files. Move each superseded Drive file into `caem-phase1a/_superseded_2026-10/` with its
     path kept.
   - Update the two Drive READMEs. The top-level one still names `vast_to_other_machine` as canonical; it
     should name `caem-phase1a/v2_1_phase1e_main`.
   - Upload the final `main.pdf`.
8. **Final check:** re-run `drive_sync_check.py`. Expect DIFFER 0 and LOCAL ONLY 0, apart from files deliberately
   kept off Drive, such as model weights.
