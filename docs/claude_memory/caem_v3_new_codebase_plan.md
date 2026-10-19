---
name: caem-v3-new-codebase-plan
description: 2026-10-19 author decision — keep final-version frozen for defense; build a new, fully organised CAEM codebase, fix all issues, real cross-benchmark verifier, re-run everything within ~2 months
metadata:
  type: project
---
Author (2026-10-19), disappointed that CAEM's contribution is weak ("a pure loss if we consider contribution"): fix every known issue, build an entirely new, fully structured codebase, re-run everything. `final-version` (now the GitHub default branch, frozen apart from read-through fixes) stays as the defense fallback. Time left: about 2 months. Goal: a genuine verifier that works beyond TriviaQA/RAG-checkable questions, plus the planned SIL fixes (phantom-context stripping, confidence weighting, cleaned-pool test, retry-when-unsure).

**Why:** v1 verifier AUROC is near chance on CSQA/StrategyQA/TruthfulQA (0.51–0.62), so SIL only helped where the verifier could check answers; matched baselines removed the headline gains.

**How to apply:** first step agreed in advice = offline verifier feasibility study on the saved, labelled answers (no GPU rental) with a go/no-go threshold before any new code or re-run. Existing evidence: answer-only token confidence AUROC (probe_answer_confidence.json) CSQA 0.70/0.74, StrategyQA 0.59/0.63, TriviaQA 0.63/0.70, FEVER 0.46/0.67, TruthfulQA 0.46/0.56 (C0/C3). Related: [[caem_final_version_plan]], [[caem_master_issues]].

**Decisions 2026-10-19 (later):**
- Research first: list questions → find real papers → store PDFs on Drive (CAEM_references) → read them with exact quotes + page (ref_check format) → experiment list, each item marked after testing. No claim from deep-research summaries without the PDF. Author upgrading to Max (deep research); will rent a GPU when needed.
- Author prefers a non-LLM verifier: an LLM judge is itself hallucination-prone, and a bigger judge raises "why not use it as the generator?". If any bigger model is used, it must check against evidence, and a bigger-model generator baseline must be added.
- Semantic entropy: v1 retired it because it clustered whole reasoning chains with MiniCheck (one-way support, not meaning equivalence); answer-level version is a candidate. Keep this distinction.
- Verifier must be benchmark-agnostic at inference (production has no dataset label); per-benchmark AUROC is only a reporting slice. v1 itself passes source_benchmark to the verifier (per-benchmark composite curves/temperatures) and to prompts — a v1 deployment gap.
- Data: dev = CAEM stream chunks (23,500) + calibration fold (7,500), split by question; test once = evaluation fold (CAEM C0–C5 + all baseline answers). Refusals set aside.
- Porting: write an inventory of the old codebase first; each ported part gets equivalence tests against v1 outputs.
- Timeline/budget (2026-10-19): defense in January 2027 (date not announced); GPU budget about $300 (Claude advised ~$100 contingency for one restart).
- Do NOT add the per-benchmark-label deployment gap to the defense thesis (author has his own answer); v3 research must solve it.
- The calibration method itself (per-signal isotonic + per-benchmark shrinkage + logistic boost fitted on the cal fold, applied at generation time) is also a research topic for v3, not assumed.
- 2026-10-19: repo `aksaN000/caem-v3` (private) created at /workspace/caem-v3; RENAMED 2026-10-19 to `aksaN000/caem-research` at /workspace/caem-research (research workspace; clean caem-v3 code repo comes at C2); **plan of record = `docs/PLAN.md` there** (read it first in any v3 session; update it in the same commit as the work). Budget raised to $400. Speed-ups are a workstream (P): v1 verifier = 108.6 of 193 h, re-ranking 68.8% of verifier time; retroverify 38 min → 16 h.
- 2026-10-19 handoff: work continues from `/workspace/caem-research/docs/PLAN.md` "Where we are" (P1 checked; P2–P5 pending; next R3 PDFs to Drive + R4 reading). Author may switch Claude account for Max; chat history will not carry over, files do.
