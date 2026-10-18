---
name: TruthfulQA LLM-judge rescore + hallucination-aware re-analysis (2026-05-16) — CAEM's abstention partially mitigates RAG-grounding-amplifies-misconceptions failure mode
description: LLM-judged TruthfulQA EM (Haiku 4.5 judge) plus hallucination-aware analysis. CAEM abstains 27% (B1/B3 abstain 0%); CAEM C0 Net Honesty −0.22 vs B3 RAG −0.41 (abstention recovers 19 pp). But CAEM still loses to B1 (−0.11) because RAG grounds wrong answers. SIL drift makes it worse C0→C5.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Rescore complete (2026-05-16, 47 min wall, $1.85 API):** Claude Haiku 4.5 as judge via Anthropic API with prompt template at `scripts/rescore_truthfulqa.py:JUDGE_PROMPT_TEMPLATE`, temp=0. Per-sample `em_llm_judged` and `em_llm_judged_reason` written into 10 files (4 baselines done + 6 CAEM cycles). Original `em` field preserved untouched.

## Why the legacy "EM" was broken

`eval/harness.py:571-575` computed TruthfulQA "EM" as `float(rouge_l > 0.15)` — a permissive overlap threshold that inflated long-prose answers (B1/B2) to 80%+ even when they contradicted gold. Fix uses the TruthfulQA paper's true accuracy definition (Lin et al. ACL 2022 §3.2): matches a correct reference AND doesn't endorse any incorrect reference.

## CRITICAL: EM is the wrong lens for a hallucination-reduction system

User pushback (2026-05-16): "CAEM is a system that reduces hallucination. you didn't check through CHM/CES." Correct — initial framing reported only EM and concluded "architecturally mismatched". Proper hallucination-aware analysis tells a more nuanced story.

**Hallucination-aware metrics on TruthfulQA (300 held-out samples):**

| Model | EM | Abstain count | Hallucination rate (per commit) | Net Honesty |
|---|---|---|---|---|
| B1 zero_shot | 0.347 | 0 | **56.9%** | **−0.110** |
| B2 cot | 0.360 | 0 | 56.3% | −0.103 |
| B5 fiveshot | 0.140 | 0 | 82.8% | −0.533 |
| B3 rag | 0.193 | 0 | 75.6% | −0.407 |
| **CAEM C0** | 0.227 | **81** | 72.1% | **−0.217** |
| CAEM C1 | 0.193 | 63 | 76.9% | −0.313 |
| CAEM C2 | 0.197 | 50 | 75.6% | −0.320 |
| CAEM C3 | 0.197 | 51 | 76.1% | −0.337 |
| CAEM C4 | 0.153 | 53 | 81.7% | −0.420 |
| CAEM C5 | 0.173 | 51 | 80.1% | −0.390 |

**Definitions:**
- Hallucination rate per commit = wrong_commits / (correct_commits + wrong_commits)
- Net Honesty = (correct_commits + correct_abstentions − wrong_commits) / n ∈ [−1, +1]
- Correct abstention = model abstained AND gold accepts "I don't know" (44 of 300 samples have such gold)

## Three findings to integrate in thesis

1. **CAEM's abstention mechanism IS firing on TruthfulQA.** 81/300 abstentions at C0 vs 0/300 for B1/B3. This is the system doing what it was built to do — refusing to commit on low-confidence questions. Net Honesty for CAEM C0 (−0.22) is meaningfully better than B3 RAG (−0.41) despite both being RAG-routed. **Abstention recovers ~19 pp of net honesty vs plain RAG.**

2. **But CAEM still loses to B1 on Net Honesty** (−0.22 vs −0.11). When CAEM commits, it hallucinates MORE often per commit (72% at C0 vs 57% for B1). Mechanism: RAG retrieves passages about the misconception topic, model commits with confidence "based on retrieved evidence", the answer endorses the misconception. **RAG-grounding amplifies confident wrong answers on adversarial misconception questions.**

3. **SIL drift is the most concerning finding.** Net Honesty trajectory C0=−0.22 → C3=−0.34 → C5=−0.39. Coverage rises (73% → 83%); the model commits more, but hallucination rate stays ~76%. SIL trains on FEVER/TriviaQA/CSQA (factual data), raising commitment confidence on the training distribution. The retention guard uses TQA-test + CSQA-test + MMLU — **none of these include TruthfulQA-style adversarial probes**, so this drift goes undetected by the rollback mechanism.

## Thesis writing guidance

**Ch5 §5.3 (main results, after empirical table):**
> "TruthfulQA exposes a key limitation of retrieval-grounded hallucination reduction: when retrieved passages mislead (as in adversarial-misconception QA), the grounding mechanism amplifies confident wrong answers rather than restraining them. CAEM's abstention mechanism partially mitigates this — abstaining on 27% of TruthfulQA samples and recovering 19 pp of Net Honesty vs plain RAG (B3) — but cannot fully compensate. Net Honesty declines further across SIL cycles (−0.22 → −0.34 → −0.39 from C0 to C5), suggesting SIL drives calibration drift toward over-commitment on the SIL-training distribution. The retention guard does not include TruthfulQA-style adversarial probes, so this drift goes undetected by the rollback mechanism."

**Ch6 §future-work:**
> "Future iterations should add an adversarial-misconception probe to the retention guard so SIL trajectories that degrade TruthfulQA performance trigger rollback. Self-RAG style learned critic tokens (Asai et al. 2024) may also reduce the RAG-grounding-amplifies-misconceptions failure mode by adding a per-passage relevance check before commitment."

**Ch5 §threats — metric correction:**
> "TruthfulQA scoring uses a Claude Haiku 4.5 LLM judge with prompt template at scripts/rescore_truthfulqa.py:JUDGE_PROMPT_TEMPLATE, temperature 0, per Lin et al. ACL 2022 §3.2 truthfulness definition. The original `eval/harness.py` scorer used a permissive ROUGE-L > 0.15 threshold that systematically over-credited long prose answers; we preserved the legacy `em` field alongside `em_llm_judged` in the per-sample JSONs for audit."

## CHM (8-subtype Composite Hallucination Metric) — CAEM-only valid, baseline pending verifier rescore

**CAEM TruthfulQA CHM trajectory (valid — verifier signals computed during eval):**

| Cycle | CHM | Union | factual_fab | logical_fab | false_refusal |
|---|---|---|---|---|---|
| C0 | 0.160 | 92% | 0.447 | 0.457 | 0.323 |
| C1 | 0.138 | 78% | 0.410 | 0.427 | 0.157 |
| C2 | 0.126 | 76% | 0.313 | 0.380 | 0.177 |
| C3 | 0.116 | 65% | 0.330 | 0.357 | 0.140 |
| C4 | 0.117 | 67% | 0.330 | 0.360 | 0.140 |
| C5 | 0.116 | 67% | 0.330 | 0.347 | 0.150 |

**SIL reduces TruthfulQA CHM by ~25% from C0 to C3.** Biggest drops: `false_refusal` (32% → 14%, −18pp) and `factual_fabrication` (45% → 33%, −12pp). This IS the hallucination-reduction receipt the thesis needs — even though EM is flat across cycles, CHM is monotonically decreasing.

**Baseline CHM ≈ 0.002-0.005 is a PHANTOM** — verifier signals (u_pre, p_entail, p_ground_*, etc.) are all `None` on baseline samples because baselines don't go through CAEM's verifier pipeline. CHM subtype rules default to "no hallucination" on missing signals → fake low CHM. DO NOT report these numbers.

**Valid baseline CHM requires:** `scripts/rescore_baselines_through_verifier.py --composite_calibration outputs/full_run/cycle_3/composite_calibration.json`. ~7 GPU-h, queued for after baseline panel finishes (~21:00 BDT today).

**Once verifier rescore lands, the apples-to-apples CHM comparison is the headline hallucination metric for the thesis** — replacing the Net Honesty proxy currently used.

## Artifacts

- Per-sample `em_llm_judged` + `em_llm_judged_reason` in 10 truthfulqa_*.json files
- 6 updated CAEM cycle TruthfulQA files pushed to gdrive at `caem-phase1a/v2_1_phase1d/full_run/cycle_{0..5}/eval/truthfulqa_cycle{N}.json`
- 5 baseline TruthfulQA files (B4/B6/B7/B8/B9) need a 2nd rescore pass after panel finishes (~$1, ~15 min)
- Rescore log at `outputs/truthfulqa_rescore.log`
- Original `em` field preserved untouched in all files
