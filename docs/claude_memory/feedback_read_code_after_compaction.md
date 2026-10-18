---
name: After compaction, READ ALL CAEM code end-to-end before answering anything
description: Hard rule — after any conversation compaction in a CAEM session, proactively read every relevant source file end-to-end before any substantive answer. On-demand reads are NOT enough; recurring conflations (Tier 1/2/3 ↔ memory/finetune/RAG, cal-fold ↔ eval-fold, α ↔ p_c) prove that.
type: feedback
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---

After any conversation compaction, the summary and memory files carry stale or
flat-out wrong attributions. The user has now caught me at least 5 times making
compaction-induced errors:

1. Attributing `u_token`/`u_dropout`/`s_avg` to Frozen Qwen instead of generator self-consistency.
2. Calling Tier 2 "RAG escalation" when Tier 2 is the fine-tuned generator and Tier 3 is RAG.
3. Conflating Step 5 transfer-eval-fold (n=500) with Step 4 stream-chunk fold (n=3000) when reading EM.
4. Conflating verifier balanced accuracy α with base-model correctness rate p_c when adjudicating theorem preconditions.
5. Saying "Tier 1 turned on at c4" while reading the cal-fold instead of the canonical n=3000 stream-chunk fold.

User's exact words: "youre so confused. read everything carefully first. your context compressed."

## Rule (strengthened)

After ANY compaction marker or session reload in a CAEM session, BEFORE answering
any substantive question (architecture, theorems, current state, fold/tier
semantics, anything load-bearing), proactively read the following files
**end-to-end, every line**:

- `/workspace/caem/scripts/run_experiment.py` — cycle orchestrator (Step 1, 2.1, 2.2-2.5, 4, 4b, 5, 6, 6b, 6c, resume logic)
- `/workspace/caem/caem/pipeline.py` — `CAEMPipeline`, tier dispatch, storage gate, `_tier1`/`_tier2`/`_tier3`
- `/workspace/caem/caem/config.py` — every threshold and locked param
- `/workspace/caem/caem/routing/router.py` — router decision + safety override
- `/workspace/caem/caem/verification/verifier.py` — `UnifiedVerifier`, 9 signals, decision tree
- `/workspace/caem/caem/sil/self_improvement.py` — `SIL.run_cycle`, MMLU retention guard
- `/workspace/caem/caem/memory/store.py` — `EpisodicMemoryStore`, novelty, retroverify
- `/workspace/caem/caem/memory/deferred.py` — `DeferredBuffer`, reconsider
- `/workspace/caem/eval/harness.py` — `EvalHarness`, per-sample record schema
- `/workspace/caem/thesis_report/chapters/chapter_4.tex` — theorems
- `/workspace/caem/thesis_report/chapters/chapter_5.tex` — receipts
- `/workspace/caem/thesis_report/chapters/chapter_1.tex` — hypothesis registry
- `/workspace/caem/thesis_report/chapters/chapter_6.tex` — conclusions

Then verify any number being quoted by grepping run.log / loading the relevant
JSON (`retroverify_cycle*.json`, `cycle_*/conformal_gate.json`, etc.).

THEN answer.

## Specific facts that must be reconfirmed each time

**Tier semantics (pipeline.py:496-571):**
- Tier 1 = memory hit, NO generation, NO Stage-5 verifier; returns `entry.answer` directly.
- Tier 2 = fine-tuned generator answers directly, no retrieval.
- Tier 3 = RAG path.
- "Escalation" = within-sample Tier 2 → Tier 3 transition on empty/failed generation (`escalated=True`).

**Fold semantics (run_experiment.py:1329, 1649-1650, 1741, 1792):**
- Step 2.1 cal fold: n=500/bench, `store_to_memory=False`, post-cycle-N-SIL model → writes `cycle_{N}/calibration/{bm}_cycle{N}.json`.
- Step 4 stream-chunk: n=3000/bench, `store_to_memory=True` → writes `eval/{bm}_cycle{N}.json`. THIS IS THE CANONICAL RUN. Memory grows only here.
- Step 5 transfer eval: n=500/bench (frozen IDs), `store_to_memory=False` → OVERWRITES `eval/{bm}_cycle{N}.json`. Watchdog snapshots Step 4 under `_streamchunk` suffix BEFORE Step 5 overwrites.

**Theorem preconditions (chapter_4.tex):**
- `thm:purity` α_store: conformal level (locked at 0.05).
- `thm:bayes-purity` α: verifier balanced accuracy (Cohen's d proxy).
- `thm:monotone` precondition: p_+ > 0.5 = base-model EM on cal fold = Tier 3 EM (NOT eval-fold overall EM).
- `thm:bayes-convergence` precondition: α > 0.5 (verifier); recurrence drives p toward 1 if p > 0.5 OR toward 0 if p < 0.5.

## Why it is NOT optional

On-demand reads (open-when-needed) repeatedly fail because compaction-induced
errors come out before I realise the question is load-bearing. Proactive
end-to-end reads catch this at session start, before any answer can go wrong.
The user has explicitly authorised the time cost: "do this whenever a session
compacts. you are troublsome to deal with when youre compacting convsersation
and miss context."

If unsure during the post-compaction read, GREP the live code rather than
reasoning from prior summary. The on-disk code is the only source of truth.
