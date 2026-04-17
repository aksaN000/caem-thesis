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

## Ch4 - deferred items
*(not yet audited)*

## Ch5 - deferred items
*(not yet audited)*

---

## Cross-chapter consistency pass (task #71) - checklist
When task #71 comes up, walk this file top-to-bottom and resolve each entry.
Every fix that edits a chapter file should be marked DONE here with the
commit/edit location before the task is closed.
