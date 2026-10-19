# Master plan and issue list (updated 2026-10-17)

## ACTIVE PLAN: minimum defensible thesis, no main re-run (author decision, 2026-10-17)
Rule: a thesis may contain a flawed component if it is described honestly; it may not contain an unfair
comparison or a description that does not match the code. So: fix comparisons and descriptions, disclose the
rest, skip the main re-run. The full re-run plan below (PART 1) is kept for the case that budget appears.

### Execution (author, 2026-10-17): one rented GPU
Runbook ready (commit 12204f2+): `scripts/rental/final_version.sh <stage>` with stages preflight, env, data, search
(+ check against run 1, gate share >= 0.95), baselines, flare, b6, gate (questions identical to CAEM's, no empty
answers), verify (C3 adapter + C3 composite as in run 1, pooled token-score path as in CAEM's main run), pack.
Upload: `scripts/rental/make_bundle.sh` (repository at HEAD + run-1 eval fold, C3 adapter/composite, splits, about
300 MB); B6 weights (5.8 GB) from Drive `CAEM_thesis/2_results_phase1e/phase1e_baselines/vanilla_ft/cycle_5` via
rclone, or uploaded. Needs `HF_TOKEN` on the rental (private index repo).
Rental spec: RTX 5090 32 GB (or larger), >= 96 GB RAM (61 GB index + 7 GB passages + models), >= 200 GB disk,
driver supporting CUDA 13 (torch 2.10+cu130).
Time estimate from run-1 logs: download 0.5–1 h; search 0.3 h; B1/B2/B5 0.3 h; B8 up to 3 h; B3/B4 TruthfulQA
0.3 h; FLARE about 2 h; B6 0.5 h; verifier pass on about 9,600 answers 4–5 h (run 1: 8.7 s/answer, of which
4.2 s search + rerank, removed by the cache except the rerank, and 3.4 s per-sample token score, pooled now).
Total about 10–12 h, about $8–12. TruthfulQA judging and all re-scoring run locally (API / CPU).
Found while preparing (2026-10-17): run 1's baseline verifier pass used the per-sample token-score path while CAEM's
main run used the pooled one (batching gate: within ±0.01); the re-run uses the pooled path.
After the rental (local): rescore_truthfulqa.py on the new baseline files, rescore_uniform.py, recompute CHM,
paired tests, tables; then chapter 5/6/1–4/appendix/abstract text, repository clean-up.
Local runs stopped and local environments/caches removed at the author's request (2026-10-17); all GPU work on the rental.

### M-A. Unfair comparisons (must fix)
- [ ] Exact passage search for the 1,500 evaluation questions (one batched pass), checked against the first
      run (re-ranking the cached top 20 must reproduce each answer's stored top 3). `scripts/precompute_passages.py`
- [x] Matched-prompt code (commit 7326a35): B1, B2, B5, B6 eval, B8 and FLARE's first draft and conclusion get
      CAEM's exact Tier 2 messages (system prompt, one worked example, task line, answer format, "Reasoning:");
      B2/B5 append "Let's think step by step."; B5 takes 5 demonstrations per benchmark (seed 42) from that
      benchmark's training split (TruthfulQA: validation), with every evaluated question and CAEM's evaluation
      fold removed by question text; demos saved to fiveshot_demos.json. Legacy prompts: matched_prompt=False.
- [ ] Re-run B1, B2, B5, B8 and B6 (saved cycle-5 weights, outputs/baselines_phase1e/vanilla_ft/cycle_5/) on all
      five benchmarks (rental).
- [ ] FLARE: its later searches use model-written text, so they cannot be pre-computed; re-run on the rental
      with live exact search.
- [x] B3/B4 empty TruthfulQA answers, cause: the first TruthfulQA batch (32 questions) hit a CUDA out-of-memory
      error (rag.gen.log, 2026-05-28 00:37) and the batch handlers stored empty (B3) or bare "Reasoning:" (B4)
      answers. Fixed (commit 397ff10): failed batch items are regenerated one at a time. All CAEM outputs and the
      other baselines have no such rows.
- [ ] B3/B4: re-run TruthfulQA on the rental (prompts unchanged; the 268 good answers must reproduce).
- [ ] Verifier pass on every re-run baseline (CHM; B8 gets its first valid CHM).
### M-B. Scoring (must fix)
- [x] TruthfulQA judge reads the final answer only; a refusal is correct only if the references include one
      (commit 7c72b27, protocol final_answer_v2; v1 verdicts kept as *_v1). v1 read the full response on CAEM's
      suppressed rows (197/300 at C0) and the refusal string on abstained rows, but the final answer for baselines.
- [ ] Re-judge: CAEM C0–C5 + Phase 1d C3 (running locally 2026-10-17, log outputs/rerun_v2/tqa_judge_v2_caem.log;
      backups outputs/rerun_v2/backup_tqa_judge_v1/); baselines after their rental re-run.
- [x] One scorer for all systems (commit 1cf7c55): eval.metrics.score_uniform, used by the harness and by
      scripts/rescore_uniform.py (writes em_final / em_lenient). CommonsenseQA: one letter, option text allowed,
      two letters wrong (the harness had no CSQA branch). TriviaQA strict = stored EM for every system (checked);
      lenient reported as an upper bound. Identical to the audit preview on 13,200 rows.
- [ ] Run scripts/rescore_uniform.py after the judge and the baseline re-runs. Downstream analysis now reads
      scripts/final_sources.py (one file per system x benchmark, label em_final; stops on anything missing):
      recompute_chm_judged.py, paired_baseline_tests.py (C0, C2, C5; B8 included), build_ch5_evidence_tables.py.
- [ ] patch_chm_judged_tables.py rewrites CHM cells only; EM cells also change (CSQA rule, judge v2): regenerate
      or extend when the numbers are in.
- [x] Router-probe id-space bug (tracker C2): fixed, re-summarised without GPU (1.0 / 1.0; not used in thesis).
- [x] CAEM scoring audit (scripts/audit_caem_scoring.py, outputs/analysis/audit_caem_scoring.json, 2026-10-17):
      closed benchmarks: no scorer damage (misses are different answers; 7 two-letter CSQA answers at C0 copy the
      "A | B | C | D | E" format line). TruthfulQA judge v2 (CAEM written, em_final): C0 0.380 -> 0.270,
      C2 0.323 -> 0.277, C5 0.300 -> 0.253; the losses are mostly refusals ("I do not know"), which TruthfulQA's own
      convention counts as truthful but not informative; v2 measures truthful and informative.
- [x] REVISED (author, 2026-10-19): delivered view follows the production response envelope (caem/production/confidence.py): only ABSTAIN (and B8 "<abstain>") is withheld; DISCARD is shown with a 'conflicting' warning; every answer carries u_stored + status tag; add a per-tag accuracy table (verified 0.75/0.66/0.67 at C0/C2/C5, provisional 0.60/0.53/0.43, conflicting 0.48/0.22/0.22, insufficient 0.37/0.27/0.31). Delivered: C0 66.1%/0.405, C2 87.7%/0.428, C5 83.0%/0.451 vs B8 89.5%/0.427, B1 100%/0.489. Baselines' verifier decisions are not charged to them (final_sources.load). CHM false-refusal (registered: abstained or discarded) unchanged.
- [x] DECIDED (author, 2026-10-17: "do what is ideal" -> both views, implemented in recompute_chm_judged.summarise and the selective table): withheld answers. EM and CHM read CAEM's raw answer even when it was withheld (ABSTAIN shows
      "I do not know.", DISCARD shows nothing). C0 withholds 335 correct answers; 517 withheld wrong answers are
      counted as hallucinations; CHM with withheld answers counted only as refusals: C0 0.155 -> 0.075,
      C2 0.127 -> 0.094, C5 0.134 -> 0.105. The current selective table counts DISCARD as answered (C0 66 % answered;
      34 % if DISCARD is withheld, wrong among answered 0.405 -> 0.294). B8 also withholds ("<abstain>": 156
      TriviaQA, 260 TruthfulQA in run 1) and is penalised the same way. Proposal: keep EM/CHM on the raw answer
      (generator view) and add a delivered view for every system (answered, wrong among answered, CHM among
      answered, refusal rate), with ABSTAIN, DISCARD and "<abstain>" as withheld.
- [ ] DISCLOSE (Ch5 + Ch6 limitations): verifier quality differs by benchmark (audit_caem_scoring.json "verifier").
      AUROC of the composite for correctness: TriviaQA 0.86 / 0.79 / 0.75 (C0 / C2 / C5); FEVER 0.64–0.67; CSQA,
      StrategyQA, TruthfulQA 0.51–0.62 (near chance: Wikipedia passages cannot confirm commonsense, multi-step yes/no
      or misconception answers). At C0 no TriviaQA answer reached the store threshold, so 124 correct TriviaQA answers
      fell to DISCARD (thresholds, not ranking; after the C1 refit DISCARD is mostly wrong answers: C2 6 right / 31 wrong).
- [x] Probe 2026-10-18 (scripts/probe_answer_confidence.py, outputs/analysis/probe_answer_confidence.json): confidence
      in the final-answer tokens only, AUROC vs correctness, C0 / C3 (stored full-response token signal in brackets):
      CSQA 0.70 / 0.74 (0.51 / 0.50), StrategyQA 0.59 / 0.63 (0.53 / 0.54), TriviaQA 0.63 / 0.70 (0.48 / 0.50),
      FEVER 0.46 / 0.67 (0.60 / 0.59), TruthfulQA 0.46 / 0.56 (0.52 / 0.56).
- [ ] Thesis wording (approved by author 2026-10-18): CAEM is "label-efficient, not label-free": about 1,500 labelled
      calibration answers fit the verifier and thresholds once; B6 needs a gold answer for every training example.
- Future work (after defense; author accepts if it helps without making CAEM much more expensive): answer-only
  confidence; semantic entropy on extracted answers (current code clusters full reasoning responses); retry-when-unsure
  then choose the higher-scored answer (CAEM never escalates now); signal clean-up + per-benchmark weights; a learned
  verifier on the same calibration labels; display thresholds by target error rate per benchmark; SIL purer pool and
  no phantom context; a contradiction signal beside MiniCheck.
### M-C. Text that does not match the code (must fix)
- [x] (DONE 2026-10-19, commit 2b57298; plus discard display fixed: code shows nothing on DISCARD) Describe as it behaved, briefly (author 2026-10-17: no fault sections): token score = generator's average
      token probability for its full response given the question (+1 limitation sentence); atomic signal reduces
      to mean grounding on short answers (existing sentence); early exit: rule + "fired on no evaluation answers
      at C0 and 3–23 per cycle afterwards"; temperature cap / safety override: honest reporting (already in ch5).
      These signals were active (cycle-zero weights: token 0.126 and half of internal 0.334; atomic 0.184), so
      they are not removed from the thesis. (Wording ACCEPTED by the author, 2026-10-17.)
- [ ] Matched-protocol claim (true after M-A), chapter 6, appendices, abstract; tables regenerated.
- [x] (DONE 2026-10-19, CORRECTED: rationalisation was OFF in the run; B6 = gold answer as target, same per-cycle questions as CAEM 2,000/2,000/700 x 5 cycles x 3 epochs = 23,500 gold pairs; the wording below was wrong) B6 description (approved 2026-10-18): "full fine-tune on gold answers to training questions, with model-written
      reasoning before each answer (rationale generated forward, or back-rationalised from the gold answer); unlike
      CAEM it sees the correct answer for every training example". Ch5 currently says "raw question-answer pairs".
      B6 trained on CAEM's Tier 2 prompt (QADataset -> build_tier2_prompt); run 1 evaluated it with the bare-question
      prompt; the re-run evaluates it with the training prompt.
### M-D. Disclosed as limitations + future work
System faults (token score scope, early exit reachability, question-match signal, atomic duplicate,
temperature cap, FEVER prefix); self-improvement erosion after C1 with its diagnosis (phantom context
3 % → 93–99 %, about a fifth of the training pool wrong) and fixes S1–S4 as future work; repeated-question
evaluation (router probe stands in); live latency (first-run batched logs 13.6–14.7 s per question, caveated).
### M-E. Trial of the self-improvement fixes: SKIPPED (author, 2026-10-17)
Self-improvement fixes S1–S4 are reported as future work with the diagnosis only.

### Uniform re-scoring preview (2026-10-17, `scripts/audit_scoring_uniform.py`, outputs/analysis/
audit_scoring_uniform.json, chm_uniform_labels_preview.json)
One fair rule for every system. FEVER, StrategyQA and TruthfulQA scoring are sound (no change for any system).
Format effects are confined to TriviaQA (sentence answers) and CommonsenseQA (letter + option text; the stored
scorer also credits the last letter of two-letter answers, 2 CAEM C0 cases). Pooled EM, stored -> uniform
(TriviaQA lenient = upper bound): CAEM C0 0.541->0.545, C2 0.544->0.550, C5 0.517->0.523; B1 0.507->0.565,
B2 0.500->0.561, B7 0.525->0.562, B6 0.490->0.523, B8 0.505->0.526. Seven-type CHM with uniform labels:
CAEM C2 0.125, C5 0.132; B1 0.141, B2 0.144, B6 0.134, B7 0.211. The matched-prompt re-run (M-A) gives the
definitive strict-format numbers. CHM breakdown under uniform labels: the remaining edge comes from fewer
factual/logical fabrications (C5 0.315/0.237 vs B1 0.341/0.285; partly by construction, since Tier 3 answers are
written from the same passages the verifier checks), not from confident confabulation (C5 0.301 vs B1 0.277;
C0 0.089 but false refusal 0.303). Selective view under uniform labels: wrong among answered C0 0.378 (66 %
answered), C2 0.418 (88 %), C5 0.444 (83 %) vs B1 0.435 (100 %): the architecture's selectivity advantage at
C0–C2 is lost by C5.

---

# APPENDIX: full re-run plan and issue list (kept for reference)

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
- [x] `C:\Users\aksan\.wslconfig`: memory=12GB, swap=16GB on `G:\WSL\swap.vhdx` (SSD). Linux sees 11 GB RAM.
- [x] Project environment `~/caem_env` with appendix-E versions: torch 2.10.0+cu130 (driver CUDA 13.0),
      transformers 4.57.6, sentence-transformers 4.1.0, faiss 1.15.1 (requirements file in Phase 1a, F3).
- [x] Models: Qwen2.5-3B-Instruct, MiniCheck-Flan-T5-Large, bge-reranker-v2-m3, all-mpnet-base-v2, in the
      Hugging Face cache on the SSD (`HF_HUB_CACHE=/mnt/g/caem_hf/hub`, set in ~/.bashrc).
- [x] GPU fit (`outputs/rerun_v2/phase0/gpu_fit_test.log`): all four models 9.86 GB; peak 10.26 GB generating
      8 questions × 3 chains; 11.85 of 12 GB with the Windows display. Local runs need batches of about 4; the
      reranker is loaded in fp32 (2.1 GB), check the first run's precision before changing it.
- Disk: the WSL disk (D:) is on a hard disk, 90–120 MB/s; the SSD (C:, G:) reads at about 245 MB/s via /mnt.
  Cold load of the 3B model: 535 s from the HDD, 27 s from the SSD. The 61 GB passage index stays on the HDD
  (one batched pass is about 10 min). **Live exact search is not practical on this laptop** (the index does not
  fit in 11 GB of RAM, so every search re-reads 61 GB); live latency (P4/B6) must be measured on a rented
  machine with ≥ 64 GB RAM or a GPU that holds the index.

## Phase 1 — Speed work (no change to results; decided 2026-10-17)
Evidence (main run log, 66,336 verified answers): the verifier took 108.6 h of about 193 h, 5.9 s per answer;
passage search + rerank 69 % (4.05 s), token score + dropout samples 16 %, chains 13 %, per-answer checks 2 %.
The search loops over questions one at a time on the CPU over the 61 GB flat index.
- [ ] **P1 Exact search only (author: no approximate index; "I don't want to miss any passage").** The
      Wikipedia index is static, so a question always gets the same passages: in research every question is
      searched once in advance (batched exact search); in deployment the same exact search runs when the
      question arrives. Pre-computation changes only when the search runs, not what it returns.
      Check: a sample of live single searches must equal the pre-computed results exactly.
- [ ] **P1b One search per question (search memo).** Tier 3 (reads top 5) and the verifier (reranks top 20)
      get the same question and the same exact index, so one search for the largest depth serves both.
      Check: memo results identical to today's two separate searches on a sample.
- [ ] **P2 Larger batches** only for the random sampling stages and pure scoring passes, and only if they pass
      the gate: every signal within ±0.01 of batch-1 scoring and no consistent shift
      (`scripts/diff_verify_serial_vs_batch.py`, about 200 answers). Batch 1 is how deployment scores a question.
- [ ] **P3 Parallel workers only in steps that do not write memory** (calibration re-score, evaluation, the
      checking part of re-verification and deferred reconsideration, probes); these are offline jobs in
      deployment too. The stream step stays sequential in batches of 32 (the one disclosed approximation).
- [ ] **P4 Stage profile** on 20 questions; **live latency** = complete path for one question at a time,
      including the live search (what a user waits for), reported separately from research throughput.

## Phase 1a — Clean results pipeline BEFORE the re-run (author, 2026-10-17)
Inventory (2026-10-17): 47 table/figure files in the thesis: 15 generated, 15 generated then hand-edited,
11 with no generator (hand-made or hand-merged), 6 hand-drawn diagrams; 97 scripts, 6 of which do work and
write files on import (aggregate_section_5_2/4/5/6/7, replay_early_stop_gate); chapter 5 prose alone holds
466 typed numbers.
- [~] R1 Script manifest (2026-10-17, branch cleanup/repo-layout: 16 one-off scripts, old runners and the approximate
      significance script moved to archive/scripts_superseded/; full pipeline/analysis manifest still to do): classify all 97 scripts as pipeline / current analysis / superseded. Move superseded
      ones to `archive/` (cannot be run by the runbook). DONE 2026-10-17 on `final-version`: the five
      aggregate_section_5_* scripts refuse to be imported; replay_early_stop_gate.py wrapped in main() (output identical).
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
- [ ] A3 Question-match signal: cycle-zero test with a rule fixed in advance: composite fitted (i) with the
      signal as is, (ii) with a rebuilt question+answer version, (iii) without it; compared on the validation
      fold (separation, balanced accuracy, stored precision); keep only if it improves the gate by the stated
      margin. (Removing the second logistic alone changes nothing for the composite.)
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

## Phase 3b — Retrieval choices (cycle-zero tests on calibration/validation folds, locked before the run)
- [ ] R-a Retrieval-trained search model vs all-mpnet-base-v2 (re-encode 21M passages, about 4–8 h on the
      3060); Tier 1 memory keeps its encoder unless the test says otherwise. Candidates from the author's
      Claude Research run.
- [ ] R-b Candidate depth before reranking: 20 / 50 / 100 (exact search).
- [ ] R-c Tier 3 reading: 3 / 5 / 8 passages, with and without reranking by the question first; guided by
      the measured rank of the first answer-bearing passage (TriviaQA, FEVER gold answers).
- [ ] R-d Verifier checks the best 3 or 5 passages.

## Phase 4 — New design elements (decided before the run)
- [ ] B3 **Repeated questions (20 %, author 2026-10-17).** From cycle 2 on, 20 % of each stream chunk re-asks
      questions from earlier chunks (exact repeats and paraphrases), inside the learning loop. A separate
      repeated-question test set is scored for CAEM and all baselines, reported next to the fresh headline.
      The old router probe becomes an optional diagnostic.
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
1. ~~Repeat share~~: 20 %, both in stream chunks and a separate test set (decided 2026-10-17).
2. ~~A3~~: decided by the cycle-zero test (decided 2026-10-17).
3. S2 training threshold and weighting.
4. Budget for Phase 7 (estimate $60–100; firm after Phases 1 and 5).
5. Retrieval-trained model candidates for R-a (from the author's Claude Research run).

## Evidence files
`outputs/analysis/`: audit_signals_baselines.json, chm_judged.json, paired_baseline_tests_judged.json,
response_relevance.json, off_topic_validation_sample.json, off_topic_judge_validation.json,
off_topic_judged.json, early_stop_gate_replay.json. Run log: `outputs/full_run/run.log` (verifier stage times).
