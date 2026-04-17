# Chapter Audit Tracking

This file catalogues audit findings across Chapters 1-5 that are deferred to
the **cross-chapter consistency pass** (task #71) because they depend on
information from more than one chapter, or because the natural fix-site is
not the chapter where the issue surfaced.

Local-correctness findings (fixable in isolation within one chapter) are NOT
tracked here - they are fixed immediately during the three-layer audit of
that chapter.

---

## Ch1 - deferred items (audited 2026-04-18)

### C1 - Three-commitments triad is restated four times in Ch1 (MEDIUM)
**Locations:** §Background line 11, §Rationale line 19, §Problem Statement
line 39, §Objective (whole section).

**Why deferred:** the right cut depends on whether Ch2/Ch3 themselves restate
the triad. If they do, the Ch1 cut needs to be larger than one paragraph; if
they don't, a lighter touch suffices. Deciding now risks either re-cutting
later or leaving duplication downstream.

**Planned fix at task #71:** once Ch2-Ch5 are audited, decide the canonical
location of the triad (likely §Background only) and reduce the others to
pointer references.

### C2 - Terminology drift: hallucination / confident hallucination / confabulation / confident-error (LOW)
**Why deferred:** the natural home for the pinned vocabulary is the Abstract,
not Ch1. Pinning it in Ch1 in isolation creates a second source of truth.

**Planned fix at task #71:** verify the Abstract pins the three-subtype
framing (confabulation / ungrounded / contradictory) once, then Ch1-Ch5 use
the pinned terms without redefining.

### A2 - §Objective preamble drops the three-subtype framing (LOW)
**Location:** Ch1 §Objective line 43 - opens with "confident hallucination"
as a monolith, losing the three-subtype commitment made in §Background.

**Why deferred:** the fix is one clause, but whether the subtype framing
needs re-stating in Ch1 §Objective depends on whether Ch2/Ch3 carry it
through. Coupled to C2.

**Planned fix at task #71:** if the Abstract pins the vocabulary, add one
clause to §Objective line 43 of the form "...reduce confident hallucination
(the three-subtype target defined in §Background)..." or remove the need
entirely by leaning on the pinned terms.

### C5 - Figure 1.2 contains numerics that belong in Ch4 (LOW)
**Location:** Figure 1.2 caption and body - rho_min = 0.93, tau_train = 0.75.

**Why deferred:** before stripping these from Ch1, I need to confirm Ch4
actually carries them. If Ch4 is missing them, stripping Ch1 deletes them
from the thesis entirely.

**Planned fix at task #71:** after Ch4 audit, if Ch4 carries the numerics,
replace Ch1 Figure 1.2's numeric labels with abstract references ("fixed
floor rho_min", "training threshold tau_train > tau_store"); cite Ch4 for
the values.

### B4 - Training-pool ID/OOD gate not mentioned in Ch1 §Constrained Self-Improvement Loop (deferred to Ch4 audit, not cross-chapter)
**Location:** Ch1 §Methodology-in-Brief §Constrained Self-Improvement Loop.
This is an implementation-level detail; natural home is Ch4 methodology.

**Planned check at Ch4 audit (task #69):** verify Ch4 §Self-Improvement
describes the ID (NQ + TriviaQA + FEVER) / OOD (TruthfulQA + StrategyQA +
ARC-C) training-pool split with reference to caem.config.TRAINING_BENCHMARKS.

---

## Ch2 - deferred items (audited 2026-04-18)

### B3 - "seven external baselines" in Ch2 vs "eight baselines" in Ch5 preface (MEDIUM)
**Location:** Ch2 §External Baselines and Competitive Positioning line 215 -
"Taken together, these seven external baselines span the space of prior
systems that share at least one mechanism with CAEM".

**Why deferred:** Ch5 preface (task #33) was corrected to "eight baselines"
including Self-RAG as a citation-level reference and STaR as an optional
ceiling. The count depends on whether STaR and Self-RAG are counted. Natural
home for the canonical count is Ch5 §5.1.Baselines (where the full lineup is
tabulated), not Ch2.

**Planned fix at task #71:** after Ch5 audit, reconcile the count. If Ch5
pins it at eight (including STaR-as-ceiling), change Ch2 line 215 to "these
eight external baselines" with a clarifying aside that STaR is an optional
ceiling. If Ch5 pins it at seven (STaR not counted), leave Ch2 as-is but add
a one-line reference to the optional STaR ceiling.

### C1 - §Preliminaries/§Review of Existing Research redundancy (MEDIUM)
**Locations:** Ch2 §Preliminaries subsections on NLI, Sentence Embeddings,
Self-Consistency, and Catastrophic Forgetting overlap topically with their
counterpart subsections in §Review of Existing Research (which re-introduces
the same concepts with citations).

**Why deferred:** any cut would need to rebalance where the architectural
claim lives (in §Preliminaries as "CAEM uses X" or in §Review as "CAEM
adopts X because prior work Y established Z"). The balance depends on
whether Ch3/Ch4 carry the architectural framing forward. If Ch4 re-explains
these concepts yet again, the Ch2 redundancy compounds across three chapters.

**Planned fix at task #71:** audit the Ch2-Ch3-Ch4 overlap once all chapters
are on the table; decide the canonical location (likely §Preliminaries as
concept-definer, §Review as literature-positioner, Ch4 as implementation)
and tighten.

### C2 - Ch1 → Ch2 transition drops without a sentence (LOW)
**Location:** Ch2 opens with §Preliminaries, but Ch1's §Methodology-in-Brief
already surfaced the architecture in detail. A reader arriving at Ch2
wonders why they are being re-introduced to LLMs and Transformers.

**Planned fix at task #71:** add one lead-in paragraph at the top of Ch2
that (a) acknowledges Ch1's architectural overview, (b) states Ch2's role
(technical preliminaries + literature survey to prepare for Ch4), and
(c) signposts that §Preliminaries defines terms and §Review of Existing
Research positions CAEM against prior work.

### A3 - Self-consistency agreement signal named $u_{\text{SC}}$ in §Preliminaries but $s_{\text{avg}}$ everywhere else (LOW)
**Location:** Ch2 §Self-Consistency Decoding line 54 defines
$u_{\text{SC}} = \frac{2}{M(M-1)}\sum \text{sim}(\mathbf{v}_i, \mathbf{v}_j)$,
but Ch2 §Hallucination Taxonomy line 21, §Verification Methods line 114,
§Confidence Estimation line 138, and Ch4/Ch5 use $s_{\text{avg}}$.

**Why deferred:** one-symbol rename, but the canonical symbol is set by
Ch4 §Verifier. If Ch4 audit reveals $s_{\text{avg}}$ is indeed canonical,
rename the Ch2 §Preliminaries definition line to $s_{\text{avg}}$ so the
symbol is consistent at its point of introduction.

**Planned fix at task #71:** rename the line-54 definition to $s_{\text{avg}}$
and update the subsequent "High consistency ($u_{\text{SC}} > 0.85$)"
sentence.

### C3 - §2.7 External Baselines and Competitive Positioning placement (LOW)
**Location:** Ch2 §External Baselines and Competitive Positioning sits
between §Self-Improvement and Continual Learning (§2.6) and §Evaluation
Benchmarks (§2.8). It discusses what Ch5 will compare against - a
forward-looking experimental-design concern inside a literature-review
chapter.

**Why deferred:** the placement is defensible (baselines draw from the
literature reviewed above) but flow-wise reads as a Ch5 preview intruding
on Ch2. Moving it to Ch5 §5.1 risks breaking citations already resolved in
Ch2. Better to decide after Ch5 audit.

**Planned fix at task #71:** after Ch5 audit, either (a) keep §2.7 in
Ch2 but add a stronger lead-in sentence that frames it as "drawing these
families into a baseline lineup instantiated in Ch5", or (b) move the
content to Ch5 §5.1 and leave a one-paragraph bridge in Ch2.

### C4 - typos surfaced during audit (LOW - polish pass)
**Locations:**
- Ch2 line 19: "etrieval" should be "retrieval"; also "rules and regulations"
  is stilted - consider "rules"
- Ch2 line 34: "rewuired" should be "required"; "t helps" should be "it helps";
  double space "and secondly,  t"
- Ch2 line 46: "retieve  sub millisecond" has a double space and missing
  hyphen - should be "retrieve sub-millisecond"
- Ch2 line 60: "Catastrophic forgetting ... suddenly lose" subject-verb
  agreement (forgetting is singular)
- Ch2 line 86: "specefic" should be "specific"; "studied by making
  ChatGPT response" should be "ChatGPT respond"
- Ch2 line 90: "designe" should be "design"
- Ch2 line 96: "Showing these intermediate steps" - unclear antecedent
- Ch2 line 148: "ventures into" should be "venture into" or reword
- Ch2 line 162: "FAISS o handle" should be "FAISS to handle"
- Ch2 line 164: "TThe" double-T
- Ch2 line 239: "various groundbreaking research. It suggests" - poor sentence
- Ch2 lines 192, 194: stray "is" / "as", preposition issues

**Why deferred:** these are proofreading, not structural. Safe to batch in a
single polish pass after all chapters are audited so the writing style is
unified across the thesis.

**Planned fix at task #71:** single typo + grammar pass across Ch2, applied
during the final polish sweep.

### C5 - Signal labels referenced before definition (LOW)
**Location:** Ch2 §Hallucination Taxonomy line 21 uses
$s_{\text{avg}}$, $p_{\text{entail}}$, $h_{\text{norm}}$,
$p_{\text{ground\_max}}$, $p_{\text{ground\_mean}}$, $p_{\text{ground\_atomic}}$,
$p_{\text{contra}}$ before any of them are formally defined. They are only
introduced later in §Verification Methods (line 114) and then again in
§Confidence Estimation (line 138).

**Why deferred:** the canonical signal table lives in Ch4 §Verifier. If
Ch4 audit keeps the table there (likely), the Ch2 mentions can stay symbolic
with a forward reference. If Ch4 reorganises, the Ch2 introduction site may
need to move.

**Planned fix at task #71:** add a footnote or parenthetical at the first
Ch2 mention (line 21) pointing to Ch4 §Verifier for the canonical symbol
definitions.

## Ch3 - deferred items (audited 2026-04-18)

### A1 - "ten cycles" vs "three cycles" reconciliation (RESOLVED this session)
**Context:** Ch3 uses "ten cycles" in seven locations (lines 10, 94, 119, 168,
221, 263, 276). During audit, A1 originally proposed changing these to "three
cycles" to match the then-pre-registered Ch1/Ch5 three-cycle framing.

**Resolution (2026-04-18):** The user reversed the cycle-count decision from
three to ten to give the memory enough horizon to approach equilibrium under
the retention guard. Ch3's "ten cycles" is now canonical; Ch1 §Scope, Ch1
§Constrained Self-Improvement Loop, Ch1 Figure 1.2, Ch2 line 202, Ch2 line
213 (Vanilla FT + EWC-only FT baselines), and caem/config.py comment block
have been updated to match. Ch5 §5.3 and the abstract to be checked during
task #70 and #71. NEXT_SESSION_PLAN and VAST_AI_DEPLOYMENT_GUIDE `--num-cycles`
flag to be flipped from 3 to 10 at task #73.

**Status:** closed. Ch3 unchanged.

### C1 - Ch1 Figure 1.1 routing labels vs Ch3 FR2 routing rules (MEDIUM)
**Locations:** Ch1 Figure 1.1 uses S-threshold bands for all three tiers
($\mathcal{S}>\tau_1$; $\tau_2<\mathcal{S}\leq\tau_1$; $\mathcal{S}\leq\tau_2$);
Ch3 FR2 line 43 uses a mixed rule (Tier 1 when S >= 0.90; Tier 2 when
sim >= 0.75; Tier 3 otherwise).

**Why deferred:** the code (`caem/routing/router.py`) matches Ch3's mixed
rule, not Ch1 Figure 1.1's uniform S-threshold depiction. The natural fix is
to update Ch1 Figure 1.1's Tier 2 label from "$\tau_2 < \mathcal{S} \le \tau_1$"
to "$\mathrm{sim} > \tau_{\text{sim}}$" so the figure matches the code. But
this is a figure-caption + TikZ-label edit on Ch1, best batched with the
cross-chapter pass.

**Planned fix at task #71:** update Ch1 Figure 1.1 Tier 2 label to reflect
the similarity-only rule (once past the tier-1 combined-score gate), and
tighten the caption so the routing rule matches the code.

### C2 - Ch3 FR6 ablation count cross-check with Ch5 (LOW)
**Location:** Ch3 FR6 line 168 describes "sixteen ablation variants";
Ch5 §5.2 preface lists the same count. The registry (caem/ablation/variants.py)
holds 16 variants. Consistent, but worth re-checking at #71 to ensure no
chapter reverts to the earlier "7-variant minimal ablation" phrasing that
appeared in old drafts.

**Planned fix at task #71:** grep all chapters for "seven ablation" /
"7 ablation" / "minimal ablation" and ensure nothing old has survived.

### C3 - Ch3 §Data hallucination-scope clause (LOW)
**Location:** Ch3 §Data commits to the five-benchmark (not TruthfulQA-alone)
hallucination-evaluation scope (task #57). Ch5 §5.3 needs the same framing.

**Planned fix at task #71:** verify Ch5 §5.3 pre-registration carries the
five-benchmark pooled-CE framing and references Ch3 §Data for the scope
definition rather than redefining.

### A2 - Ch3 routing-threshold rationale (LOW, polish)
**Location:** Ch3 FR2 explains the λ=0.70 weighting but does not motivate
τ_1=0.90 or τ_sim=0.75 beyond "chosen so that...". A half-sentence on why
these specific numbers (tight similarity bar for high-quality hits, combined
score weighted toward similarity over stored quality) would tighten the
rationale without lengthening the chapter materially.

**Planned fix at task #71:** add a one-sentence rationale after the FR2
threshold definition pointing to the calibration procedure in Ch4.

## Ch4 - deferred items (audited 2026-04-18)

### Local-correctness fixes applied during audit (not deferred)
- **Line 1114 ten-cycle straggler:** "three to five are evaluated end-to-end
  in Chapter~5 given compute budget; the choice of evaluated cycle count is
  reported in Chapter~5 and does not change the loop mechanism" rewritten to
  "all ten of which are evaluated end-to-end in Chapter~5; the per-cycle
  trajectory reported there is the full ten-cycle sweep." This matches the
  canonical ten-cycle decision (A1 resolution) already applied to Ch1/Ch2/Ch3.

### Ch4 Layer-A/B/C summary (all clean, no deferred items)
- **Layer A (concept validity):** Pre-Routing, Episodic Memory, Three-Tier
  Router (OR-condition τ_safe=0.60 + combined score S with α=0.70 + tier
  thresholds τ_tier1=0.90, τ_tier2_sim=0.75), Unified Verifier (nine signals
  + composite weights 0.30/0.15/0.15/0.15/0.15/0.10 + early-exit gate
  u_internal≥0.70 ∧ p_ground_max≤0.20), Four-Outcome Storage (τ_veto=0.30,
  τ_store=0.65, τ_defer=0.45, τ_abs=0.20, τ_train=0.75), Retroverify
  (τ_retro=0.50 three-way cascade with upward-only rule, abort-skip),
  Deferred Reconsider (N_buf=10,000, TTL T=2, novelty filter on promotion),
  Self-Improvement (L2 anchor λ=0.01, r_gen=0.10, ρ_min=0.93, MMLU n=200,
  epochs E=3, LR 1e-5, batch 16), Theoretical Analysis (Purity, Monotonicity,
  Convergence theorems) - all mathematically sound and internally consistent.
- **Layer B (concept-to-code alignment):** Verified against verifier.py,
  router.py, memory store, self-improvement loop. All thresholds, signal
  weights, gate logic, and hyperparameters match code 1:1.
- **Layer C (narrative flow within Ch4):** Section ordering follows the
  pipeline stage sequence without gaps; Implementation hparams table at
  line 1380 correctly tags "Cycles C = 10 [DES]" (matches canonical decision);
  Theorem statements reference their own equations cleanly.

### B4 cross-check follow-through (training-pool ID/OOD gate, from Ch1)
Ch4 §Self-Improvement Loop (lines 1073-1148) does not explicitly call out
the ID (NQ + TriviaQA + FEVER) / OOD (TruthfulQA + StrategyQA + ARC-C)
training-pool split by name; the text cites $\mathcal{D}_{\text{gen}}$
generically. This is acceptable at Ch4 granularity - the split is a dataset-
configuration concern that Ch5 §5.3 pre-registration should pin. Moved to
task #70 (Ch5 audit) as the natural home.

### Minor polish candidates (not worth a deferred-item entry)
- Line 1369 tags λ=0.01 as [LIT] citing EWC; arguably [DES] is more accurate
  since CAEM uses the L2 simplification rather than full Fisher-weighted EWC.
  Defensible as [LIT] ("EWC approximation"); leave as-is unless a reviewer
  flags it.

## Ch5 - deferred items (audited 2026-04-18)

### Local-correctness fixes applied during audit (not deferred)

**MAJOR LAYER-B FIX: Ablation variant registry rewritten to match code.**
The previous Ch5 §Ablation Methodology described 16 ablation variants
(e.g. `no_verification`, `no_retrieval`, `no_l2_anchor`, `no_abort_guard`,
`no_nli`, `no_selfcons`, `no_entropy`, `no_routing`, `no_cold_start`,
`no_memory_prune`, `no_mcdropout`, `lower_u_threshold`) that bear almost
no resemblance to the actual `caem/ablation/variants.py` registry. This
was a stale-spec artefact from task #36 (expected-effects prose written
before the registry was reverted in task #47). Fix applied during this
audit:
- Rewrote Table `tab:ablation-registry` (Ch5 §Ablation Methodology)
  to list the 16 code-registered variants exactly, grouped by
  `mechanism_tag` (reference / verification / calibration / safety_gate /
  self_improvement / routing / retrieval).
- Rewrote the four "expected effects and falsifiable predictions"
  paragraphs (§5.2 Self-improvement / Verification-and-gating /
  Calibration-and-signal-mix / Safety-gate / Routing / Retrieval-and-memory)
  to reference the correct variant names and describe their actual
  config mutations, with predicted $\Delta$CES bands tied to the
  actual mechanism each variant ablates.
- Updated two trailing variant-name references: the "diagnostic per-cycle
  CES trace" list in §Metric reported, and the "macro outcomes" paragraph
  in §Interpreting the full ablation pattern.
- All stale variant names (`no_verification`, `no_retrieval`,
  `no_l2_anchor`, `no_abort_guard`, `no_routing`, `no_cold_start`,
  `no_memory_prune`, `no_mcdropout`, `no_nli`, `no_selfcons`, `no_entropy`,
  `lower_u_threshold`) have been removed from Ch5; grep confirms zero
  remaining references.

### Ch3 correction surfaced during Ch5 audit (applied, not deferred)
Ch3 Ablation registry table (line 234) previously claimed `No grounding`
zeros "four" external grounding weights; the code's `_mut_no_grounding`
zeros exactly three ($p_{\text{ground,mean}}$, $p_{\text{ground,atomic}}$,
$p_{\text{entail}}$). Fixed to "three" with the three signals enumerated.

### Ch5 Layer-A/B/C summary (all other content clean, no deferred items)
- **Layer A (concept validity):** Experimental setup (six factual-QA
  benchmarks + MMLU retention), Metrics (CES geometric-mean composite,
  pooled confident-error rate, four-outcome data-purity breakdown),
  Statistical significance (McNemar + Holm-Bonferroni over $m=8$
  baselines with $\dagger/\star$ two-marker scheme + $\Delta \text{EM}
  \ge 0.02$ practical-significance floor), Implementation Details
  (N=10 cycles, n_eval=500/benchmark, n_cal=500, n_purity=500,
  n_MMLU=200, n_screen=1500 all disjoint non-overlapping splits),
  Per-cycle calibration protocol (T re-fit by LBFGS on NLL each cycle,
  only runtime-active calibration surface), Fixed-threshold + post-hoc
  sensitivity sweep, Cold-start memory (447 episodes from FEVER/TriviaQA/NQ).
  All internally consistent and consistent with Ch3/Ch4.
- **Layer B (concept-to-code alignment):** Variant registry now matches
  `caem/ablation/variants.py` 1:1 (after the fix above). Metric formulas
  match `eval/metrics.py`. Threshold values ($\tau_{\text{store}}=0.65$,
  $\tau_{\text{train}}=0.75$, $\lambda_{\text{route}}=0.70$, $u_{\text{pre}}$
  floor $=0.60$, $\rho_{\min}=0.93$) match config. Compute budget
  ($\sim 18$ GPU-h main + $\sim 16$ GPU-h baselines + $\sim 24$ GPU-h
  screening) matches the observed per-cycle timing envelope.
- **Layer C (narrative flow within Ch5):** §Experimental Setup →
  §Ablation Methodology → §Main Results → §Ablation Results →
  §Chapter Summary. The pre-registration content (§5.2 expected-effects
  and §5.3 expected-results) is placed before the results sections so
  that the results discussion can be graded against pre-committed
  predictions rather than narrated as post-hoc rationalisation.
  Internal cross-references (`\cref{tab:headline}`, `\cref{thm:purity}`,
  etc.) resolve cleanly.

### B3 cross-check follow-through (baseline count, from Ch2)
Ch5 preface line 349 and line 581 both say "eight external baselines"
(B1-B8) with STaR as an optional ceiling reference. This confirms the
eight-count is canonical at Ch5 and that Ch2's "seven external baselines"
(B3 deferred item) needs to be reconciled to "eight" at task #71.

### C3 cross-check follow-through (hallucination-scope clause, from Ch3)
Ch5 §Benchmarks line 58-65 and §Metrics line 147-194 both commit to the
pooled six-benchmark confident-error framing with the ID/OOD split
(ID: NQ + TriviaQA + FEVER; OOD: TruthfulQA + StrategyQA + ARC-C) rather
than the earlier TruthfulQA-alone scope. This confirms Ch3 C3's deferred
check passes; no further action needed at #71 beyond cross-reference
verification.

### B4 cross-check follow-through (training-pool ID/OOD gate, from Ch1 → Ch4)
Ch5 §Metrics line 182-194 pins the ID (NQ + TriviaQA + FEVER) / OOD
(TruthfulQA + StrategyQA + ARC-C) split with the Stage~5 verifier
training-pool gate implicit in the ID component. The pooled-CE ID/OOD
decomposition is the natural home Ch1 B4 was deferred toward; no
additional Ch1 edit needed at #71 beyond verifying the forward pointer
from Ch1 §Constrained Self-Improvement Loop to Ch5 §Metrics.

### Minor Ch5 robustness note (not worth a deferred-item entry)
§Statistical significance line 281 says "raw test count is $8 \times 7
= 56$ pairwise McNemar tests per cycle". The "$\times 7$" counts six
factual-QA benchmarks plus MMLU. MMLU comparisons against baselines
are only meaningful for the two trained baselines (B6 Vanilla FT,
B7 EWC-only FT); for B1-B5, B8 the MMLU row is the pristine Flan-T5
score which is identical to \caem{}'s Cycle-0 MMLU and carries no
cross-method signal. The $8 \times 7 = 56$ framing is conservative
(over-counts by $\sim 6$ tests that will have degenerate $p=1.0$); the
Holm-Bonferroni correction remains valid either way. Worth a sentence
of clarification at task #71 but not a deferred-item entry in itself.

---

## Cross-chapter consistency pass (task #71) - checklist
When task #71 comes up, walk this file top-to-bottom and resolve each entry.
Every fix that edits a chapter file should be marked DONE here with the
commit/edit location before the task is closed.
