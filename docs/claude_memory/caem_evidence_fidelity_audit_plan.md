---
name: CAEM evidence-fidelity audit tables — deferred to Ch1–6 rewrite pass
description: During the post-Phase-1a Ch1–6 rewrite, add hand-audited critical-case tables to Ch5/Ch6 proving (a) multi-gate architecture stops hallucinations from reaching users, (b) model improves monotonically each cycle. Backs the "no hallucinated answer propagates" claim and the "model always improves" claim.
type: project
originSessionId: 2f06df76-4467-4646-9e32-0a9234c27ac4
---
On 2026-04-22 during Step 6 (cold-start seed) the user raised that the
verifier's ~4% store-gate evidence-hallucination pass-through rate
observed in Profile v8 (51-episode FEVER smoke at τ=0.45) does not
actually become a user-facing hallucination rate, because CAEM has a
4-layer defense:

| Gate | Field | Default | Role |
|------|-------|---------|------|
| 1    | store_threshold (auto-calibrated at P70 after Cycle-0) | ~0.50–0.65 | Entry to memory |
| 2    | min_u_stored_for_training | **0.75** | Entry to SIL fine-tuning pool |
| 3    | tier1_combined_threshold (0.7·sim + 0.3·u_stored ≥ threshold) | **0.90** | Tier-1 cached retrieval |
| 4    | retroverify_prune_threshold | 0.50 | Cycle-boundary re-score and prune |

All 3 Profile-v8 evidence-hallucination cases (#2, #24, #48) had
u_stored between 0.48 and 0.70 — so all are filtered by gate 2
(τ_train = 0.75) and essentially all by gate 3 (which requires
sim ≥ 0.97+ at those u_stored levels). The user-facing hallucination
rate is effectively zero even though the store-gate rate is ~4%.

**Commitment — execute during the Ch1–6 thesis rewrite pass
(post-Phase-1a completion), NOT in the current execution session.**
The goal is to turn the "4% store miss" weakness into a strength by
hand-auditing critical cases and tabulating the multi-gate filtration.

**Three tables to produce:**

**Table 5.A "Multi-gate hallucination filtration"** — hand-audited
~20 critical cases per cycle covering the u_stored distribution
(top, middle, bottom). Columns:
- Question (truncated)
- Generated answer (truncated)
- Ground-truth verdict (correct / evidence-hallucinated / wrong)
- u_stored
- Passes gate 2 (y/n)
- Passes gate 3 at realistic sim (y/n)
- Retroverify disposition at next cycle (KEEP / DOWNGRADE / PRUNE)
- Net: does this episode EVER surface as a user-facing answer? (y/n)

Target demonstration: **no hallucinated episode surfaces as user-facing
output** (the net column is all 'n' for misclassified cases).

**Table 5.B "Monotonic model improvement per cycle"** — per-cycle
deltas proving the self-improvement loop converges upward:
- Cycle c (1..N, where N is equilibrium cycle via Upgrades 1–3 fit)
- EM (mean across benchmarks)
- F1
- Mean u_stored on STORE decisions
- Stored-episode purity (hand-audited sample)
- MMLU retention ratio
- Verdict: Δ vs previous cycle (should be non-negative on EM/F1/
  purity; MMLU retention should stay ≥ 0.93 forgetting_tolerance)

Target demonstration: **strictly monotonic non-decreasing EM and
purity across cycles 1..N** (equilibrium visible as the plateau).

**Table 5.C "Human-vs-CAEM-verifier agreement"** — gold-standard
verifier-quality measurement most SIL papers omit. Sample ~200
episodes across u_stored range and all 3 training benchmarks
(FEVER + TriviaQA + NQ, ~67 per benchmark). For each:
- Human label ∈ {correct, conclusion-right-evidence-wrong, wrong-answer, uncertain}
- Verifier decision ∈ {STORE, DEFERRED, ABSTAIN, DISCARD}
- u_stored scalar

Metrics to report:
- 4×4 confusion matrix (human × verifier)
- Precision of verifier STORE vs human "correct" label (target ≥ 0.95)
- Recall of verifier STORE vs human "correct" (report whatever lands)
- Cohen's κ for human-verifier agreement (target ≥ 0.6 "substantial")
- AUC of u_stored as a detector of human "correct" (target ≥ 0.85)
- Per-benchmark breakdown (FEVER / TriviaQA / NQ)

These three headline numbers position CAEM's 9-signal verifier as
substantially better-calibrated than a MiniCheck-only baseline
(MiniCheck paper reports single-signal κ ≈ 0.40–0.55 typical).

Total hand-audit work: ~200 labels → ~6–8h of labelling effort.
Persist as `outputs/audit/human_vs_verifier_labels.json`.

**Table 5.D "Failure-mode subtype breakdown of CAEM-caught episodes"**
— defends the defense-critical claim "we catch hallucinations, not
just low-confidence outputs." Piggyback on Table 5.A/5.C labelling:
for each already-audited episode, add one more column tagging the
failure subtype:

- Confabulation (invented fact) → HALLUCINATION (strict)
- Ungroundedness (answer not in any retrieved passage) → HALLUCINATION
- Evidence-hallucination (right conclusion, fabricated citation) → HALLUCINATION
- Factual error (wrong date/name/number, plausible) → HALLUCINATION-adjacent
- Over-hedging ("not enough info" when info exists) → CONSERVATIVE FAILURE
- Off-topic (answers a different question) → RELEVANCE FAILURE

Target: >= 70% of catches classified as hallucination-strict
(first 3 rows). The remaining ≤ 30% are desirable-filtration but
not hallucinations per se. This table shuts down the "you don't have
a hallucination detector, you have a confidence filter" reviewer
critique by showing the majority of catches are genuine hallucinations.

Signal-to-subtype mapping (defend the "primary-target" claim):
- 6 of 10 signals directly target hallucination: s_avg, h_norm,
  p_ground_max, p_ground_mean, p_ground_atomic, p_contra
- 2 target hallucination-adjacent uncertainty: u_dropout, p_entail
- 2 target non-hallucination failures: u_token (hedging),
  q_a_relevance (off-topic / Goal-2 sample-②)

Include this mapping as a figure or inline table in Ch5 §5.X so the
"verifier is primarily a hallucination detector" framing is
empirically and architecturally justified.

Marginal additional work for Table 5.D: ~1h (adds one column to
already-audited episodes).

**Table 5.E "Per-episode-type cycle-resolution trajectory"** — defends
the claim "effective verifier decision accuracy converges to 100% at
steady state on in-memory episodes" (the sharpened form of the user's
intuition "verifier should be 100% accurate").

Trace ~15 canonical episodes from Table 5.A across cycles 1..c*.
Categories (aim for 2-3 episodes per category):
- TP-high (correct + high u_stored): stable KEEP throughout
- TP-moderate (correct + moderate u_stored 0.55-0.70): retroverify
  PROMOTES past τ_train=0.75 by cycle M
- TP-low (correct + DEFERRED, 0.45 ≤ u_stored < τ_store): deferred
  buffer reconsideration promotes to STORE at cycle M
- FP-moderate (hallucination at u_stored 0.55-0.70): retroverify
  DOWNGRADE → PRUNE over 1-2 cycles
- FP-high (rare confabulation at u_stored 0.75+): retroverify at
  cycle+1 catches; represents 1-cycle-of-pollution worst case
- FN (correct + u_stored < τ_defer=0.45): DISCARDED, unrecoverable

Columns per episode:
- episode_id, category
- u_stored[c] for c in 1..N (trajectory)
- retroverify_disposition[c] (KEEP / DOWNGRADE / PRUNE / RECONSIDER-PROMOTE)
- final_fate (TRAINED / STORED-ONLY / PRUNED / CONSOLIDATED-AWAY / LOST)

Target demonstration: by cycle c* (equilibrium via Upgrades 1-3 fit),
every in-memory episode correctly classified — TPs trained, FPs
pruned, duplicates consolidated. The FN category has no trace because
it was never stored; documented as the honest limitation.

**Defensible claim from Table 5.E:** "Effective decision accuracy on
in-memory episodes approaches 100% at steady state via
retroverify + deferred reconsideration + consolidation acting as a
joint corrective mechanism over cycles 1..c*." The qualifier
"in-memory" is essential — false-negatives below τ_defer are
unrecoverable and this limitation must be explicitly acknowledged in
the thesis. Do NOT write the bare universal "verifier is 100%
accurate" — that's false at Cycle 1 (~95% precision observed) AND
false at steady state (FN are permanently lost).

Marginal additional work for Table 5.E: ~2h (requires reading multiple
retroverify JSONs across cycles and assembling per-episode trajectory).

**Total revised audit effort (Tables 5.A through 5.E): ~12-16h.**

**Data source:** after Step 7 completes (estimated wall-clock ~100–170
GPU-h from 2026-04-22 launch), extract from:
- `outputs/full_run/cycle_N/memory_store.{faiss,meta}` — stored episodes
- `outputs/full_run/retroverify_cycleN.json` — gate-4 dispositions
- `outputs/full_run/experiment_summary.csv` — EM, F1, MMLU retention
- `outputs/full_run/cycle_N/equilibrium_fit.json` — c* prediction
  (shipped in commit d341622, wired into run_experiment.py cycle loop)

Persist the hand-audited classifications as
`outputs/audit/evidence_fidelity_manual.json` before rendering Tables
5.A/5.B. ~20 cases × 10 cycles = ~200 manual classifications; estimate
4–6 h of hand-audit work.

**Chapter placement:**

- **Ch5 §5.X new subsection "Evidence-fidelity audit"** — describes
  the methodology and presents Tables 5.A, 5.B, and 5.C. Cites
  Upgrades 1–3 equilibrium fit for the cycle-range selection.
- **Ch6 §Discussion** — 1-paragraph integration of the 4-gate
  architectural-defense framing, citing Tables 5.A, 5.B, and 5.C as
  quantitative backing for (1) "no hallucinated answer propagates to
  user" (Table 5.A gate trace), (2) "model monotonically improves per
  cycle" (Table 5.B EM/purity deltas), and (3) "9-signal verifier
  substantially outperforms single-signal baselines" (Table 5.C
  human-gold agreement).
- **Ch1 §Motivation** — brief forward-reference to §5.X if needed to
  preempt reviewer concerns about hallucination rate.

**Non-goal of this plan:** making the verifier store-gate itself
tighter. That remains calibrated at P70. The thesis story is that
multi-gate + retroverify is what closes the loop, not a single
stricter store threshold.

**Status:** queued for the Ch1–6 rewrite pass. Log entry
`branch_C_log.md` 2026-04-22 17:30 BDT captures the full detail.

**CRITICAL — thesis claim discipline (don't overclaim):**

The tables let us claim a rigorously-bounded version of "training
data is pure." The sloppy universal claim "100% pure always" is NOT
defensible (it requires exhaustive-audit-or-proof). The defensible
phrasing is the conjunction:

1. Empirical: "0 of N audited evidence-hallucinations entered the
   training pool" (N = Table 5.A row count)
2. Architectural: "τ_store < τ_train guarantees every u_stored < 0.75
   miss is filtered from training by construction" (trivially from
   config)
3. Residual-risk: "High-u_stored evidence-hallucinations (u_stored ≥
   0.75 despite 9 signals agreeing) are addressed by cycle-boundary
   retroactive re-verification (§4.7); Table 5.B shows retroverify
   catches [X] such cases across cycles 1..N"

When the user says "our training data is 100% pure always," the
response is: yes, in the sense of (1) + (2) + (3) conjoined — do NOT
write the bare universal claim in the thesis. Reviewers will ask for
a counterexample proof and the bare claim will collapse.
