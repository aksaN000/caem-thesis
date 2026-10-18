---
name: CAEM Phase 1 Cycle-0 audit findings (2026-04-25, n=3500)
description: Empirical audit of broken-composite Cycle-0 across 7 benchmarks; the evidence base that drove Phase 2 patches.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
Phase 1 audit was the deep cross-benchmark teardown of the post-A2-v2
broken-composite Cycle-0 eval (archived at
outputs/archive/pre_a2v3_broken_composite_2026-04-25/ — full 7-benchmark
JSONs + manifest). Findings:

## Per-signal Cohen's d × 7 benchmarks (em=1 vs em=0 in eval fold)

```
signal              fever  triviaqa   NQ      truth   strategy  ARC      ASQA
u_stored           +0.380  +1.273  +0.773   +0.265   +0.316  +0.156   +0.480
p_ground_mean      +0.548  +0.931  +0.712   +0.244   +0.320  +0.159   +0.624
p_ground_max       +0.106  +1.037  +0.714   +0.215   +0.322  +0.172   +0.790
p_entail           +0.219  +0.931  +0.280   +0.250   +0.226  +0.456   -0.549
p_ground_atomic    +0.548  +0.931  +0.712   +0.244   +0.320  +0.159   +0.624  (= p_ground_mean; 100% fallback)
q_a_relevance      -0.577  +0.498  +0.748   +0.261   +0.188  +0.082   +2.958
u_internal         +0.092  +0.506  +0.246   +0.284   -0.144  +0.122   +0.178
s_avg              -0.098  +0.306  +0.224   -0.418   -0.162  -0.103   +0.208
u_dropout          -0.098  -0.523  -0.316   -0.117   +0.127  -0.084   +0.144
h_norm             -0.082  -0.330  -0.165   +0.118   -0.201      nan  +0.513
u_token            -0.031  -0.050  -0.058   +0.257   -0.157  +0.102   +0.542
```

Key sign flips: q_a_relevance (FEVER negative, others positive),
p_entail (ASQA negative, others positive), s_avg / h_norm / u_token
(inconsistent across benchmarks). u_dropout consistently weak/negative.

## Per-benchmark base EM and STORE-decision precision

```
benchmark         EM      STORE n    STORE prec   atomic fallback
fever             0.428    31         0.806        100%
triviaqa          0.272     5         0.600        100%
NQ                0.156     9         0.333        100%
truthfulqa        0.378     3         0.667        100%
strategyqa        0.588     7         1.000 (n=7)  100%
arc_challenge     0.736    24         0.880        100%
asqa              0.014     3         0.000 (n=3)  100%
```

Atomic 100% fallback even on ASQA confirms the broken-composite era's
atomic mechanism was structurally degenerate everywhere.

## Behavioural failure modes (the EM-recovery problem)

```
benchmark         empty %    "I do not know" %     u_dropout >= 0.8
fever             43.6        21.6                  65.0
triviaqa          43.0        40.4                  89.6
natural_questions 44.4        38.8                  97.8
truthfulqa        53.0        37.0                  99.0
strategyqa        17.4        69.8                  45.0
arc_challenge     12.0        55.0                   0.0
asqa              44.8        45.2                  97.6
```

The base model produces ~30% pooled em-correct answers (~2500 out of
9000 SIL queries/cycle), but the broken composite recovers only ~10% of
these em-correct samples (~240/cycle). Three bottlenecks identified:
(1) prompt non-compliance (43% empty), (2) over-abstention on open-QA,
(3) atomic decomposition degenerate.

## Conformal split-CP empirical findings (offline, training-only calibration)

Phase 1.5c proper-discipline test (calibration on FEVER+TriviaQA+NQ
training samples only, applied to held-out training + transfer):

```
operating point     ID storage rate    transfer storage rate
@80% precision      3.3%               1.4%
@70% precision      varies             11.2%
```

The 80% target IS reachable on both ID held-out and pooled transfer
under the cal-prob composite + conformal gate, but at small storage
volumes. NQ and ASQA are mathematically capped below 70% by sparse em=1
counts (NQ: 78 em=1 in 500; ASQA: 7 em=1 in 500). All other benchmarks
reach 70% empirically.

## Why this matters for the thesis

The audit data is the empirical foundation Phase 2 patches were
designed against. Each Phase 2 patch's docstring cites these specific
numbers (e.g., "43.6% empty rate observed" for the prompt fix; "100%
atomic fallback" for the length-gate scope; "q_a_relevance d=-0.577 on
FEVER" for the per-signal isotonic auto-handling). Phase 4 thesis
artifacts will surface these tables in Ch5 §Verifier Calibration
Diagnostic.
