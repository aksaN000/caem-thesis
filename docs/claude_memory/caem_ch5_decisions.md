---
name: caem_ch5_decisions
description: "Chapter 5 author decisions 2026-10-16: title drops 'Progressive'; headline = B1→C0→C5 decomposition; H1 supported, H2/H5 unsupported, H4 partial; sweep re-run numbers only; phantom-context paragraph; selective-risk view."
metadata:
  type: project
---

Full text in `defense/thesis_report/ref_check/SESSION_START.md` ("Chapter 5 decisions"). Key numbers:
- EM B1 0.507 / C0 0.541 / C2 0.544 / C5 0.517; CHM 0.120 / 0.135 / 0.107 / 0.109. Training-panel EM C0 0.571, C1 0.584, C5 0.549.
- Confident confabulation and false refusal are verifier-defined (eval/metrics.py: wrong & u ≥ 0.50 (eval/metrics.py default); correct & ABSTAIN/DISCARD) → trade off with the gate's operating point.
- Selective view: coverage/error-among-answered B1 1.00/0.493; C0 0.661/0.381; C1 0.856/0.421; C2 0.877/0.425; C5 0.830/0.451.
- Gate balanced accuracy pooled: C0 0.753 (in-sample), C1 0.616, C2 0.668, C3 0.669, C5 0.661.
- Title now "Confidence-Aware Episodic Memory with Self-Improvement for Hallucination Reduction in Large Language Models".

Aksan's questions answered: the purity theorem is simple and about the pool, not the model; its value is a testable precondition and a design target. The measurement is computed correctly, but the subtypes depend on the system's own verifier, so the selective-prediction view (Kamath et al., Geifman and El-Yaniv) is added as the standard, operating-point-independent complement.

Related: [[caem_three_headline_decomposition]], [[caem_phantom_context_finding]], [[caem_early_stop_gate_artefact]].

**Open (2026-10-16):** (1) B8 CHM in the thesis is from the OLD broken SE run (pre-rerun backup, rescored 2026-05-29); the corrected run was never verifier-rescored (u_stored None) → GPU rescore or report unavailable — ask Aksan. (2) CHM "significance" is an approximate unpaired bootstrap (no Holm) in scripts/baseline_sig_tests_v2.py; a true paired test from per_sample_chm is feasible for B1–B7 (validated against committed CHM). (3) Repo plug-and-play clean-up planned as a phase after the chapter pass. WIP pushed on review/ch5 (no PR yet).

**B8 resolved (2026-10-16):** group has no budget; B8 CHM reported as unavailable (Drive searched incl. folder 1NDl20m8…: only old-run with_chm files; phase-2 rerun rescored B6/B7 only). Counts without B8: C5 CHM 19 wins/2 defeats of 30, C2 18/3. Next ch5 step: paired CHM test (per_sample_chm, Wilcoxon + Holm) + appendix D fix, then Aksan's read-through → PR.

**2026-10-17 read-through decisions (author):** H5 REMOVED (ch1+ch5; supervisor never saw it; four hypotheses now) with Tier 1 framing: built for repeated questions in deployment, study uses fresh questions, probe tests it, EM and CHM expected to improve with repeat share (expectation, not measured). Early-stop mis-wiring disclosure removed (see [[caem_early_stop_gate_artefact]]). Ch5 body tables 39→21 (rest in appendix B/C/D), all \small, no \resizebox. CHM significance now paired Wilcoxon+Holm (`scripts/paired_baseline_tests.py`): C5 21/3 of 35, C2 24/4 of 35 (ROUGE label). Off-topic subtype zero = double sigmoid (tracker C14); inverting doesn't help (reranker flags short answers). Verifier-free CHM check: C5 0.067 vs B1 0.079 (26/4 of 35). PENDING author decision: TruthfulQA CHM label ROUGE vs Haiku-judged (tracker D11; judged → B1 0.137, C0 0.134, C2 0.110, C5 0.117; TQA CHM defeats vanish). Format check (CPU): lenient answer-line match adds a flat ~3 points on TriviaQA at every cycle, so format does not hide a post-C2 gain.

**2026-10-17 CHM label decision (author):** "truthfulqa always works with haiku, any other thing used is stale method" — CHM now uses em_llm_judged on TruthfulQA everywhere (scripts/recompute_chm_judged.py → outputs/analysis/chm_judged.json). New pooled CHM: B1 0.137, C0 0.134, C2 0.110, C5 0.117; B6 0.122 nearest baseline; FLARE 0.198. Paired CHM C5 19/1, C2 23/1 (only defeat B6 CSQA). Narrative: C0 CHM level with B1; refusal/confab trade nearly cancels; net C0→C5 CHM gain from fewer fabrication/evasion. Ch6 must use these numbers.

**2026-10-17 off-topic investigation:** author wants a real off-topic value and accepts CHM changes ("final thesis is meant to have changes"). Reranker on bare answer / rewritten sentence / full response all fail (full-response vs 200 Claude labels: precision 0.12, recall 0.40). Only an LLM judge works: `scripts/judge_off_topic.py` (Haiku, wrong answers on TriviaQA+TruthfulQA, ~4,950 calls, ~$2, ~2 h) — BLOCKED: ANTHROPIC_API_KEY on machine invalid (rotated). Plan: judge → validate vs 200 labels (15 OFF) + author checks 40 → if valid, off-topic enters CHM (stays /8); else CHM /7. Run signal not harmful (AUROC bare 0.798 vs full 0.745 at C0 TriviaQA). Found D15: B5 answers FEVER labels on 235/300 TriviaQA, 289/300 TruthfulQA (FEVER demos). Reranker env: ~/caem_reranker_env (torch cu121, ST 3.3.1, transformers 4.46.3, anthropic).

**2026-10-17 topic 1 DONE** (commit 79194b4): signal renamed question-answer match q_match, all chapters; CHM = mean of 7 (B1 0.156, C0 0.153, C2 0.126, C5 0.133). **Audit findings pending author discussion** (tracker C16, C17, D16–D19): u_token scores whole response under bare prompt (near-useless); early exit could not fire at C0 (0/1500); TriviaQA strict EM undercounts verbose baselines (B1 lenient 0.553 vs strict 0.293) — threatens B1→C0 EM headline; TruthfulQA judge credits 30% of refusals even without no-comment references (CAEM C0). B5 rerun not started (design of demonstrations to agree). New API key in ~/.bashrc (pasted in chat → rotate later).
