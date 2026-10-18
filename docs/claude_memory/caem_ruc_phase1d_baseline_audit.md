---
name: Phase 1d B1 → CAEM C5 baseline audit (Haiku-judged TQA) — RUC on/off ablation receipt
description: The canonical pre-RUC empirical baseline against which Phase 1e (CAEM-with-RUC) will be compared. Three-contrast decomposition (B1→C0 architecture, C0→C5 SIL, B1→C5 full system) + per-bench breakdown + 7-baseline comparison panel. All TruthfulQA EM uses em_llm_judged (Haiku) per `feedback_truthfulqa_llm_judge_em.md`. Use this directly for Ch6 §sec:ruc-ablation tables and the Phase 1e contrast.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
## Measurement protocol

- TruthfulQA EM: `sample.em_llm_judged` (Haiku rescore). For per-sample CHM on TruthfulQA, override `sample.em` with `em_llm_judged` before calling `composite_hallucination_metric`.
- All other benches (FEVER, TriviaQA, CommonsenseQA, StrategyQA): default `sample.em`.
- CHM: canonical 8-subtype equal-weighted mean from `eval.metrics.composite_hallucination_metric`.
- Eval pool: 300 samples per bench, cycle_0 (B1 baselines) and cycle_5 (CAEM trajectory close), from `gdrive:caem-phase1a/v2_1_phase1d/full_run/eval/`.
- Cycle 5 is Phase 1d's last successful cycle (cycle 4 retention guard fired and rolled back; cycle 5 closed with C3 weights).

## Three thesis contrasts (pooled, Haiku-judged TQA)

| Contrast | Δ EM | Δ CHM | Interpretation |
|---|---|---|---|
| B1 → CAEM C0 (architecture only) | **−3.67pp** | **+16.1%** | Adding three-tier router + verifier + RAG regresses both axes |
| C0 → CAEM C5 (SIL only) | +0.47pp | **−18.4%** | SIL recovers most CHM regression, marginal EM |
| **B1 → CAEM C5 (architecture + SIL)** | **−3.20pp** | **−5.2%** | Phase 1d C5 LOSES 3pp EM but WINS 5% CHM vs base Qwen |

## Per-bench B1 → C5 contrast (the RUC-on-off receipt)

| Bench | B1 EM | C5 EM | Δ EM | B1 CHM | C5 CHM | Δ CHM | Class | Phase 1e routing target |
|---|---|---|---|---|---|---|---|---|
| FEVER | 0.453 | 0.507 | **+5.4pp** ✓ | 0.158 | 0.122 | **−22.8%** ✓ | RAG-helpful (win) | preserve via 56% T3 |
| TriviaQA | 0.293 | 0.437 | **+14.4pp** ✓ | 0.172 | 0.123 | **−28.5%** ✓ | RAG-helpful (win) | preserve via 67% T3 |
| CommonsenseQA | 0.703 | 0.600 | **−10.3pp** ✗ | 0.098 | 0.133 | **+35.7%** ✗ | RAG-hurt (loss) | neutralise via 92% T2 |
| TruthfulQA (Haiku) | 0.433 | 0.213 | **−22.0pp** ✗ | 0.131 | 0.137 | +4.6% ≈ | RAG-hurt (loss, EM only) | neutralise via 65% T2 |
| StrategyQA | 0.650 | 0.617 | −3.3pp ≈ | 0.114 | 0.121 | +6.1% ≈ | RAG-hurt (slight loss) | neutralise via ~70% T2 |

**Pattern:** Phase 1d wins decisively on 2 RAG-helpful benches (FEVER, TriviaQA), loses decisively on 2 RAG-hurt benches (CSQA, TruthfulQA), ties StrategyQA. The pooled −3.2pp EM gap vs B1 is the bench-asymmetric loss the RUC is designed to fix.

## Phase 1d CAEM C5 vs full baseline panel (pooled, Haiku-judged TQA)

| Baseline | Δ EM | Δ CHM | Who wins? |
|---|---|---|---|
| B1 zero_shot | −3.20pp | −5.2% | CAEM CHM ✓, EM ✗ |
| B2 cot | −2.47pp | −7.3% | CAEM CHM ✓, EM ✗ |
| B3 rag | +1.80pp | −14.0% | CAEM ✓✓ |
| B4 cot_rag | +0.87pp | −9.8% | CAEM ✓✓ |
| B5 fiveshot_cot | +6.93pp | −5.8% | CAEM ✓✓ |
| B6 flare | +18.33pp | −20.0% | CAEM ✓✓ |
| B7 semantic_entropy | −3.20pp | −4.0% | CAEM ≈ (EM ✗, CHM ≈) |

**Pattern:** Phase 1d C5 leads the panel on CHM (best at 0.127, edges B7 at 0.133), and beats all RAG-augmented baselines on both axes. Loses to no-RAG baselines (B1, B2, B7) on EM, ties B7 on CHM.

## Per-subtype Phase 1d C0 → C5 trajectory (canonical 8-subtype, Haiku-judged TQA)

| Bench | CC C0/C5 | FF C0/C5 | LF C0/C5 | FR C0/C5 (false refusal) | DE C0/C5 |
|---|---|---|---|---|---|
| FEVER | 0.087/0.330 | 0.477/0.327 | 0.300/0.250 | 0.290/0.060 | 0.153/0.007 |
| TriviaQA | 0.000/0.173 | 0.463/0.347 | 0.423/0.383 | 0.400/0.083 | 0.030/0.000 |
| CommonsenseQA | 0.387/0.393 | 0.267/0.290 | 0.383/0.377 | 0.013/0.007 | 0.003/0.000 |
| TruthfulQA (Haiku) | computed via em_llm_judged override |  |  |  |  |
| StrategyQA | 0.073/0.267 | 0.380/0.360 | 0.240/0.287 | 0.453/0.053 | 0.140/0.003 |

**Key trade-off:** Phase 1d traded **false refusal** (−77 to −88% across benches) for **confident confabulation** (+264 to +inf% relative). The model stopped over-abstaining and converted refusals into confident-wrong outputs. Aggregate CHM still improved because false_refusal + defensive_evasion dropped harder than CC rose. Pooled CC: 0.110 (C0) → 0.251 (C5) = +128% relative, the empirical receipt that motivates the RUC.

## The RUC's mechanism — "wins preserved, losses neutralised"

Aksan's articulation (2026-05-19): *vanilla CAEM had hurtful negative outcomes on RAG-hurt benches and helpful positive on RAG-helpful; RUC turns hurtful into neutral while keeping helpful as positive.*

| Query type | Phase 1d (no RUC) | Phase 1e (with RUC) |
|---|---|---|
| RAG-helpful | T3 + RAG → positive | T3 + RAG → positive (preserved) |
| RAG-hurt | T3 + RAG → negative | T2 parametric → neutral (rescued) |
| Marginal | mostly T3 → small ± | RUC picks per-query → small ± |

**Asymmetric value:** wins are preserved (no downside risk), losses are neutralised (no further damage). Net pooled = win + neutral instead of win + loss.

## Phase 1e predictions (RUC-on ablation expected outcome)

Predicted under the asymmetric-rescue mechanism, conditional on the trajectory completing without retention abort:

| Metric | Phase 1d C5 | Phase 1e C5 predicted | Direction |
|---|---|---|---|
| Pooled EM (Haiku TQA) | 0.475 | **0.500 - 0.520** | close 3-pp gap |
| Pooled CHM | 0.127 | **0.115 - 0.130** | hold or improve |
| Pooled B1 → C5 ΔEM | **−3.20pp** | **0 to +2pp** | **rescue the gap** |
| Pooled B1 → C5 ΔCHM | −5.2% | **−5 to −10%** | hold or extend lead |
| TruthfulQA EM | 0.213 | **0.330 - 0.430** | recover most of the 22pp loss |
| TruthfulQA CHM | 0.137 | **0.110 - 0.140** | flat or improve |
| CSQA EM | 0.600 | **0.680 - 0.705** | close 10pp loss via 92% T2 |
| CSQA CHM | 0.133 | **0.105 - 0.120** | undo +35% regression |
| StrategyQA EM | 0.617 | **0.640 - 0.660** | small recovery via 70% T2 |
| Pooled Confident Confabulation rate | 0.251 | **0.140 - 0.180** | shave back ~30-45% |
| Pooled False Refusal rate | 0.072 | 0.080 - 0.100 | small rebound (T2 abstains more) — acceptable cost |

## How to use this in the report

**Ch5 §sec:results — main table**: load Phase 1e C5 from the trajectory once it finishes, place side-by-side with this Phase 1d baseline. Same protocol (Haiku TQA, 8-subtype CHM, 300 samples per bench).

**Ch5 §sec:adj-cal-eval-gap**: cite the per-subtype CC trade-off (+128% pooled) as the failure mode the calibration gap couldn't catch.

**Ch6 §sec:ruc-ablation — the headline contrast**: RUC-off (this Phase 1d baseline) vs RUC-on (Phase 1e trajectory result). The expected mechanism is "wins preserved, losses neutralised"; the empirical receipt is the per-bench B1→C5 ΔEM/ΔCHM table.

**Ch6 §sec:concl-headline**: "*The Retrieval Utility Classifier closes a 3pp pooled EM gap and a 22pp TruthfulQA EM gap that the unconstrained three-tier architecture introduces, by routing RAG-hurt queries to the parametric tier where their behavior reverts to base-model neutral instead of RAG-amplified negative.*"

**Threats / future work**: small false-refusal rebound predicted as the RUC's only structural cost. If observed, register as "v1.x → v2 trade: confident-wrong reduction comes at the cost of slight over-abstention recovery; adaptive RUC retraining per cycle could close this in v3."

## Pre-registered claims (lock before Phase 1e closes)

1. **Pooled B1 → Phase-1e-C5 ΔEM ≥ 0pp** (i.e., RUC closes Phase 1d's −3.2pp gap).
2. **Pooled CHM ≤ Phase 1d C5's 0.127** (RUC preserves the CHM lead).
3. **TruthfulQA EM (Haiku) ≥ 0.330** (RUC recovers >50% of the 22pp gap).
4. **CSQA EM ≥ 0.680** (RUC closes >75% of the 10pp gap).
5. **Pooled CC rate ≤ 0.18** (RUC shaves back >30% of the +128% CC regression).
6. **No retention guard abort across all 10 cycles** (RUC's bench-conditional routing prevents the C4-type rollback that fired in Phase 1d).

Any of (1)-(6) failing is a real result that needs to be discussed in Ch5 §sec:disc-threats, not papered over.
