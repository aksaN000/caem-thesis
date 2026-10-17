# Master plan and issue list (rewritten 2026-10-17)

This file is the single checklist for the re-run. It replaces the scattered limitation lists. Tracker ids
(C/D/T/S, in REPO_CLEANUP_TRACKER.md) are cross-referenced. Add every new finding here.

## Principles (author decisions, 2026-10-17)
1. Fix everything that makes the current system correct and lets self-improvement help; then re-run the
   experiment once and rewrite the thesis once.
2. No multiple seeds, no second backbone, no extra baselines, no long-horizon runs (avoided items below).
3. Every change is decided before the re-run; nothing is tuned on the evaluation fold.
4. Fair comparison: every baseline gets CAEM's exact question message; the same scorers and judge for all.
5. TruthfulQA is always judged by Haiku; CHM is the mean of seven failure types.

---

# PART 1 — THE PLAN (phases in order)

## Phase 0 — Machine setup (local)
- [x] `C:\Users\aksan\.wslconfig`: memory=12GB, swap=16GB on `F:\WSL\swap.vhdx` (C: has 36 GB free).
- [ ] Author: `wsl --shutdown` from PowerShell, reopen; Claude checks `free -g` shows about 12 GB.
- [ ] Download MiniCheck-Flan-T5-Large and all-mpnet-base-v2; check they load on the RTX 3060 next to the
      3B model (target about 10 GB of 12 GB GPU memory).
- Already on disk: Qwen2.5-3B-Instruct, bge-reranker-v2-m3, passage index (61 GB) and passages (7 GB).

## Phase 1 — Speed work (no change to results)
Evidence (main run log, 66,336 verified answers): the verifier took 108.6 h of about 193 h, 5.9 s per answer;
passage search + rerank 69 % (4.05 s), token score + dropout samples 16 %, chains 13 %, per-answer checks 2 %.
The search loops over questions one at a time on the CPU over the 61 GB flat index.
- [ ] **P1 Retrieval cache.** The search uses only the question, and every question of the run is known in
      advance (stream chunks, folds, cold start, probes; about 30,000). Search them all in one batched exact
      search; save question → top 20 passages. The verifier (top 20) and Tier 3 (top 5) read from it.
      Check: compare with one-by-one search on 500 questions (expect identical lists apart from exact ties).
      Deployment still searches live; the cache is an experiment shortcut only.
- [ ] **P2 Larger batches** only for the random sampling stages (3 chains, 5 dropout samples) and pure scoring
      passes (reranker, MiniCheck, token score with right padding). Greedy answer generation and atomic
      decomposition keep their validated settings (the code documents that batched atomic decomposition
      changed the facts). Check with `scripts/diff_verify_serial_vs_batch.py` on about 200 answers: accept a
      stage only if every signal moves less than 0.01 and in no consistent direction.
- [ ] **P3 Parallel workers only in steps that do not write memory:** calibration-fold re-score, evaluation,
      the checking part of memory re-verification and deferred reconsideration (changes applied afterwards in
      one serial pass), retention probes. Two workers fit on a 32 GB GPU (about 11–12 GB each).
- The stream step stays sequential in batches of 32, as in the last run, so routing and Tier 1 see memory as
  a deployed system would. Fine-tuning stays a single job.
- [ ] **P4 Stage profile:** time 20 questions stage by stage on the 3060 to replace estimates with numbers.

## Phase 1a — Clean results pipeline BEFORE the re-run (author, 2026-10-17)
Inventory (2026-10-17): 47 table/figure files in the thesis: 15 generated, 15 generated then hand-edited,
11 with no generator (hand-made or hand-merged), 6 hand-drawn diagrams; 97 scripts, 6 of which do work and
write files on import (aggregate_section_5_2/4/5/6/7, replay_early_stop_gate); chapter 5 prose alone holds
466 typed numbers.
- [~] R1 Script manifest (2026-10-17, branch cleanup/repo-layout: 16 one-off scripts, old runners and the approximate
      significance script moved to archive/scripts_superseded/; full pipeline/analysis manifest still to do): classify all 97 scripts as pipeline / current analysis / superseded. Move superseded
      ones to `archive/` (cannot be run by the runbook); guard the six import-time scripts with a main block.
- [ ] R2 Data vs words: every data table is produced by a script as a bare tabular from the run's JSON; the
      caption, label and float placement live in the chapter file (hand-written words are legitimate there).
      The hand edits made so far (merges, layouts, dropped columns, notes below tables) are re-implemented as
      generator code, so nothing is lost and nothing is edited by hand again.
- [ ] R3 One command, `make_results`, regenerates every data table and figure from the outputs; a check mode
      regenerates into a temporary folder and diffs against the committed files (must be identical).
- [ ] R4 Numbers in prose: the same command writes a macro file (e.g. `\NumBoneEM`) from the JSONs; the
      Phase 8 rewrite cites macros instead of typed numbers, so no number can go stale.
- [ ] R5 Hand-drawn diagrams (chapter 4, architecture) stay hand-made but are reviewed against the Phase 2
      design (e.g. the dropped atomic signal) and listed in the manifest.
- [ ] R6 Runbook: exact commands for every stage of the re-run; only those commands are used.

## Phase 2 — System fixes
- [ ] A1 Token score: answer tokens only, scored under the prompt that produced them.
- [ ] A2 Confabulation early exit: re-check that it can fire after A1; thresholds stay as specified.
- [ ] A3 Question-match signal: remove the second logistic; build the claim from question + answer.
      Decision needed: keep (fixed) or drop.
- [ ] A4 Drop the atomic-fact signal (identical to mean grounding on short answers).
- [ ] A5 Pre-routing temperature cap / safety floor: fix so the override can work.
- [ ] A11 Strip the benchmark prefix (FEVER boilerplate) before Tier-1 embedding.
- [x] A12 Early-stop check wiring (fixed 2026-10-16).
- [ ] A13 Remove dead signals (semantic entropy path, contradiction, two unused diagnostics).

## Phase 3 — Let self-improvement help (special care)
- [ ] S1 (A6) Training chains: strip or rewrite "the context ..." references (Tier-2-style targets).
- [ ] S2 (A7) Purer training pool: higher training threshold and/or confidence-weighted loss
      (SIL_POISON_DRIFT_FINDINGS); report pool precision before each fine-tune.
- [ ] S3 (A8) Retention probes on every benchmark, including TruthfulQA and CommonsenseQA.
- [ ] S4 Training targets in the evaluation's short-answer format.
- [ ] S5 Purity gain from A1/A2 (confident, unsupported answers discarded before storage).
- Decisions needed: training threshold value; weighting scheme; which probes and floors.

## Phase 4 — New design elements (decided before the run)
- [ ] B3 **Repeated questions.** From cycle 2 on, a stated share of each stream chunk re-asks questions from
      earlier chunks (exact repeats and paraphrases). A separate repeated-question test set is scored for CAEM
      and all baselines. The fresh evaluation fold stays the headline. Decision needed: repeat share
      (suggested 20 %).
- [ ] B6 Live-search latency at batch size 1, per tier (the cache would otherwise understate deployment cost).
- [ ] A9 (optional) One frozen-composite control cycle, if time allows.

## Phase 5 — Local mini-run (RTX 3060, free)
- [ ] Cycle zero + one cycle on a subset, with all fixes. Purpose: check that the self-improvement fixes help
      (no phantom context, pool precision, retention) and confirm timing before renting.

## Phase 6 — Baselines and scoring (local GPU / CPU / API)
Run after Phase 2 freezes CAEM's Tier 2 question message.
- [ ] C1 Re-run B1, B2, B5, B8 and FLARE's first draft with CAEM's exact question message on all five
      benchmarks (the thesis claims matched protocol; today it holds only for B3/B4).
- [ ] C2 B5 demonstrations taken from each benchmark's own training split.
- [ ] C3 Re-generate B3/B4's 32 empty TruthfulQA answers.
- [ ] C4 Verifier pass on B8's corrected run (CHM).
- [ ] C5 B6: check the weights are on disk; matched prompt; retention measurement.
- [ ] D1 TriviaQA: after C1, re-check strict vs lenient gap for every system; judge if still needed.
- [ ] D2 TruthfulQA judge reads the final answer only; a refusal counts only if the references include one.
- [ ] D9 Fix the router-probe id-space bug.
- Baselines are re-scored through the new verifier, so every system is measured by the same instrument.

## Phase 7 — Full re-run (rented)
- Cycle zero (calibration + 100-variant sweep), cycles 0–5, evaluation, probes, repeated-question set.
- Device: RTX 5090 (best value; ~$0.64/h last time), 2 workers in the non-writing steps. The 61 GB index is
  not needed on the rented machine once the cache exists.
- Estimate: about 50–70 h on one RTX 5090 (~$35–50) for the main run; with cycle zero and margin about
  $60–100 (baselines run locally). Firm number after Phase 1 profiling and the Phase 5 mini-run.

## Phase 8 — Results and write-up
- [ ] Run `make_results` (Phase 1a) on the new outputs; check mode must pass.
- [ ] One rewrite of chapters 1–6, appendices and abstract (category E), then the chapter 6 future-work list.
- [ ] Repository clean-up (category F).

---

# PART 2 — FULL ISSUE LIST WITH STATUS
Status: PLAN (phase n), DONE, AVOIDED (limitation / future work), INHERENT.

## A. System faults
| # | Issue | Status | Ref |
|---|---|---|---|
| A1 | Token score covers the whole response under a bare prompt; near-useless (median 0.04–0.06 at C0) | PLAN 2 | C16 |
| A2 | Early exit could not fire at C0 (0/1500), at most 23/1500 later | PLAN 2 | C17 |
| A3 | Question-match signal: bare answer, double logistic, cannot detect off-topic | PLAN 2 | C14, C15 |
| A4 | Atomic-fact signal identical to mean grounding | PLAN 2 | C9 |
| A5 | Pre-routing temperature at its cap; safety override nearly inactive | PLAN 2 | C7 |
| A6 | Phantom context: 93–99 % of zero-shot answers cite a context after C1 | PLAN 3 (S1) | C12 |
| A7 | About a fifth of the training pool is wrong; accuracy erodes after C1–C2 | PLAN 3 (S2) | D6 |
| A8 | Retention guard misses TruthfulQA and CommonsenseQA losses | PLAN 3 (S3) | ch5 |
| A9 | Composite refit every cycle confounds the monotonicity check | PLAN 4 (optional) | ch5 |
| A10 | Verifier checks only the final answer, not reasoning steps | AVOIDED | memory |
| A11 | FEVER prefix dominates Tier-1 embeddings | PLAN 2 | memory |
| A12 | Early-stop check mis-wired | DONE (code) | C1 |
| A13 | Dead signals still computed | PLAN 2 | C8, C11 |

## B. Design limits
| # | Issue | Status |
|---|---|---|
| B1 | Single seed, single backbone | AVOIDED |
| B2 | Six cycles only | AVOIDED |
| B3 | Tier 1 never exercised | PLAN 4 (repeated questions) |
| B4 | Classifier ablation confounded by other corrections | AVOIDED |
| B5 | Classifier not tested on a held-out benchmark | AVOIDED |
| B6 | Per-tier latency not measured | PLAN 4 (live search, batch 1) |
| B7 | Deferred-buffer survival not recorded; short horizon | AVOIDED (record pass counts if cheap) |
| B8 | Corpus ceiling not measured | AVOIDED |
| B9 | Calibration fold 500 per benchmark | AVOIDED |

## C. Baselines
| # | Issue | Status | Ref |
|---|---|---|---|
| C1 | Prompt mismatch for B1, B2, B5, B8, FLARE (all five benchmarks) | PLAN 6 | D16 |
| C2 | B5 FEVER-only demonstrations (FEVER labels on 235/300 TriviaQA, 289/300 TruthfulQA) | PLAN 6 | D15 |
| C3 | B3/B4: 32 empty TruthfulQA answers | PLAN 6 | D18 |
| C4 | B8 has no valid CHM | PLAN 6 | D1 |
| C5 | B6: weights to check, retention unmeasured, copied cycle-3 file, no answer line | PLAN 6 | D4, D19 |
| C6 | STaR, Self-RAG not run | AVOIDED | ch5 |

## D. Scoring
| # | Issue | Status | Ref |
|---|---|---|---|
| D1 | TriviaQA strict match penalises sentence answers (B1 0.293 vs 0.553 lenient) | PLAN 6 | D16 |
| D2 | TruthfulQA judge credits 30 % of refusals without a reference refusal | PLAN 6 | D17 |
| D3 | Off-topic cannot be measured | DONE (excluded; detection = future work) | C14, C15 |
| D4 | CHM partly measured by CAEM's verifier | DONE (verifier-free check) | ch5 |
| D5 | CHM used ROUGE on TruthfulQA | DONE (Haiku) | D11 |
| D6 | CHM significance approximate | DONE (paired Wilcoxon + Holm) | C3 |
| D7 | B4 EM dropped empty answers | DONE | D12 |
| D8 | Cycle-zero files carry the legacy composite | removed by the re-run | D2, D3 |
| D9 | Router-probe id-space bug | PLAN 6 | C2 |

## E. Thesis text (after the re-run, Phase 8)
E1 token score and early-exit descriptions; E2 matched-protocol claim; E3 chapter 6 revision; E4 appendices
(Drive names, ARC rows, appendix C, appendix G citations); E5 abstract (709 words); E6 bib key
`fu2025collapse`. Done already: question-match reframe, CHM as seven types, Haiku TruthfulQA, paired test,
H5 removal, early-stop wording, chapter 5 table restructure.

## F. Repository (Phase 1a for F1, F2, F5; the rest in Phase 8)
F1 hand-edited tables (T1–T11); F2 scripts that overwrite on import (S1); F3 requirements file (S2);
F4 evidence in an ignored folder (S5); F5 superseded scripts and tables (S3, S6, T6, T8); F6 stale comments
and dead config (C4, C5, C6, C10, C13); F7 cold-start as one stage (D9); F8 Drive-only receipts, tags (D8, S4).

## G. Inherent limits of scope (reported)
G1 short-answer questions only; G2 memory tested to 12k entries; G3 verifier-filtered training converges to
the verifier's knowledge; G4 consistent misconceptions are missed; G5 lucky guesses on binary tasks;
G6 low-rank adaptation forgets with training; G7 conformal guarantees dropped under shift; G8 512-token
judge input.

## Open decisions for the author
1. Repeat share for B3 (suggested 20 %).
2. A3 question-match signal: keep (fixed) or drop.
3. S2 training threshold and weighting.
4. Budget for Phase 7 (estimate $60–100; firm after Phases 1 and 5).

## Evidence files
`outputs/analysis/`: audit_signals_baselines.json, chm_judged.json, paired_baseline_tests_judged.json,
response_relevance.json, off_topic_validation_sample.json, off_topic_judge_validation.json,
off_topic_judged.json, early_stop_gate_replay.json. Run log: `outputs/full_run/run.log` (verifier stage times).
