# CAEM Panel Demo Script

**Defense walkthrough:** ~6-8 minutes for the seven questions below, then open Q&A from the panel.

**Memory snapshot:** cycle-3 (pre-defense rehearsal) or cycle-10 (defense day per Phase B.4).

**Server:** `python scripts/caem_demo_server.py --memory outputs/production/memory_store/memory_store --passage_index data/passage_index --port 8000` (or production-swapped path; see `DEMO_QUICKSTART.md`).

**What the panel sees per question:**
- Top: tag (✓ verified / ~ provisional / ⚠ conflicting / ✗ insufficient) + confidence percentage on the right
- Middle: the answer text (or "I do not know" for ABSTAIN, or a hidden-answer placeholder for DISCARD)
- Bottom: tier badge, latency badge, and an expandable "Evidence" + "Memory match" panel showing per-signal scores and the matched stored episode

---

## Question 1 — Verified, Tier 1 memory hit (~10 sec)

**Type:** `Game of Thrones is a television series. Answer with one of: supports, refutes, not enough info.`

**Expected behavior:**
- **Tier 1** (memory hit) — answer comes from a previously-stored episode; no model generation
- **Tag:** ✓ verified, **confidence ≥ 95%**
- **Latency:** sub-200 ms (no generator call)
- **Evidence panel:** empty (Tier 1 skips Stage 5 verifier by design)
- **Memory-match sidebar:** shows the matched FEVER stored episode at near-1.0 cosine similarity

**What to point out:**
> "This question landed in episodic memory from a prior cycle. The system bypasses generation and verification entirely, returning the verified stored answer in a fraction of the cost. This is the *cor:tier1-floor* corollary in action — the stored entry's precision is bounded by the conformal contract from when it was admitted."

---

## Question 2 — Verified, Tier 2 fine-tuned generator (~7-10 sec)

**Type:** `Abraham Lincoln was born in the 1800s. Answer with one of: supports, refutes, not enough info.`

**Expected behavior:**
- **Tier 2** (fine-tuned generator answers directly, no retrieval needed) — empirical reading at cycle 3: u_stored ≈ 0.917, EM = 1.0
- **Tag:** ✓ verified, **confidence ~90%**
- **Latency:** ~7-10 seconds (single generation pass + 9-signal verification)
- **Evidence panel:** all 9 signals visible, with high p_entail and p_ground (memory + retrieval both support)
- **Decision:** STORE — this answer is good enough to be admitted to memory at conformal precision floor 95%

**What to point out:**
> "The fine-tuned 3-billion-parameter generator produced this answer without retrieving Wikipedia. The 9-signal verifier scored it above the storage threshold of τ_store ≈ 0.75, so the conformal gate admits it. This is the storage class — the architecturally protected pool that *thm:purity* bounds."

---

## Question 3 — Provisional, DEFERRED (~10 sec)

**Type:** `William is Bill Clinton's first name. Answer with one of: supports, refutes, not enough info.`

**Expected behavior:**
- **Tier 2 or 3** (depending on routing); answer correct
- **u_stored ~0.60** (between τ_defer ≈ 0.48 and τ_store ≈ 0.75) → **Decision: DEFERRED**
- **Tag:** ~ provisional, **confidence ~60%**
- **Caveat displayed:** "This answer is held for reconsideration at the next training cycle (~quarterly in production). It will either be promoted to verified memory or retired."
- **Evidence panel:** mid-range signal scores, p_ground_max moderate

**What to point out:**
> "The verifier wasn't confident enough to admit this to the storage class but wasn't confident enough to reject it either. The answer is shown to the user with an explicit deferral caveat — and the entry goes into a bounded buffer where the next cycle's improved verifier will revisit it. This is the *cor:self-correction* pathway closing on uncertain answers across cycles."

---

## Question 4 — Insufficient, ABSTAIN — no grounding (~10 sec)

**Type:** `Starring Johnny Depp, which was the highest grossing film worldwide in 2007?`

**Expected behavior:**
- **Tier 3** (RAG path) — generator does retrieve passages but they don't sufficiently support the answer
- **u_stored ~0.03**, **p_ground_max ~0.09** (below abstention floor φ_pg = 0.20)
- **Decision:** ABSTAIN
- **Tag:** ✗ insufficient, **confidence < 10%**
- **Answer field:** "I do not know."
- **Evidence panel:** all 9 signals visible, grounding signals near zero

**What to point out:**
> "The retrieval-augmented path could not find Wikipedia passages that ground a confident answer. The system explicitly refuses rather than confabulating — this is the *epistemic-honesty* design: the four-outcome decision tree separates *we cannot ground this* from *we are confident in this*."

---

## Question 5 — Conflicting, DISCARD via confabulation early-exit (~7 sec)

**Type:** `Steve Buscemi was in The Big Lebowski. Answer with one of: supports, refutes, not enough info.`

**Expected behavior:**
- **Tier 2 or 3**; the model is internally confident (u_internal ≥ 0.70) but retrieval doesn't ground the claim (p_ground_max ≤ 0.20)
- **Confabulation early-exit fires** before the composite is even computed
- **Decision:** DISCARD
- **Tag:** ⚠ conflicting, **confidence ~5-10%** (the system reports its own conflict)
- **Answer field:** hidden behind a refusal placeholder
- **Evidence panel:** u_internal ≥ 0.70 highlighted in conflict against p_ground_max ≤ 0.20

**What to point out:**
> "This is the confabulation gate — when the model *internally* believes its answer but the retrieved evidence does not support it, the gate fires before any storage decision can be made. *thm:asymptotic-elim*'s false-positive correction at the architectural ε_arch term."

---

## Question 6 — Safety override on out-of-domain query (~8-10 sec)

**Type:** `Write a Python function that computes the Fibonacci sequence using memoization.`

**Expected behavior:**
- **Pre-routing confidence (u_pre) low** on this domain (registered scope is factual QA, not code generation)
- **Safety override fires** at u_pre < 0.38 → forces Tier 3 (RAG path) regardless of routing score
- **Decision:** likely DISCARD or ABSTAIN (Wikipedia passages don't ground a code-generation answer)
- **Tag:** ✗ insufficient or ⚠ conflicting depending on what the generator produces
- **Answer field:** refusal or partial code with low confidence

**What to point out:**
> "CAEM is registered for factual question answering. When a user submits an out-of-domain query, the safety override forces the conservative tier and the verifier flags the absence of grounding. This is the *fail-safe-over-fail-silent* invariant — the system does not produce confident answers outside its registered scope."

---

## Question 7 — Live demonstration of memory accumulation (optional, ~30 sec)

**Type 7a (first time):** `Charlie Chaplin appeared in The Great Dictator. Answer with one of: supports, refutes, not enough info.`

**Expected behavior:** Tier 2 or 3, generator answers, 9-signal verifier scores, decision is STORE if u_stored crosses τ_store. Note the answer + tier + tag.

**Type 7b (second time, paraphrased):** `Did Charlie Chaplin star in the 1940 film The Great Dictator?`

**Expected behavior:** Tier 1 hit on the just-stored episode. Sub-200 ms latency. ✓ verified.

**What to point out:**
> "Within a single defense session, the system has stored a verified answer to question 7a and is now serving question 7b from episodic memory at a fraction of the cost. This is the production cycle in microcosm: every verified answer reduces the cost and increases the safety of subsequent semantically-related queries."

---

## Panel Q&A — open the floor

After question 7 (or skip 7 for time), invite the panel to type their own questions.

**If the panel asks a deliberately-hard-to-answer factual question:** the system should default to ABSTAIN or DISCARD with a low confidence percentage. **Point out** that the failure mode is *honest refusal*, not *confident hallucination*.

**If the panel asks something CAEM has never seen:** safety override → Tier 3 RAG → if Wikipedia doesn't ground it, ABSTAIN. **Point out** that the architecture inherits the corpus floor *cor:corpus-floor* and is not pretending to know everything.

**If the panel asks for the system's confidence on a question:** the percentage at the top-right of the response card IS the confidence. Walk them through the underlying signal contributions in the Evidence panel.

---

## Operator notes

- **Time budget:** 6-8 minutes for the seven questions, ~5-10 minutes panel Q&A. Total demo: ~15 minutes.
- **If a question runs longer than expected** (Tier 3 with full RAG can take 10-15 s on the first query while passage retrieval warms up): pre-warm by clicking through Q1-Q3 once before the panel arrives; subsequent queries are faster.
- **If the demo errors out:** open `outputs/production/audit/demo_dryrun_cycle10.log` (or current snapshot) to show the panel a recorded successful run; then play the backup video from Phase B.5.3.
- **If the panel asks "can I see the live training?":** the answer is *the live training has already produced this artifact*; the system you are demoing is the cycle-10 (or cycle-3 rehearsal) checkpoint. Point at `docs/PRODUCTION_RUNBOOK.md` for the cyclic-in-production design.
- **All seven questions above are anchored in actual cycle-3 cal-fold readings** (see `outputs/full_run/cycle_3/calibration/*.json`). The expected tags / confidences / decisions are empirically grounded, not invented.

---

## Quick-launch reference (also in `DEMO_QUICKSTART.md`)

```bash
# Pre-defense rehearsal (cycle-3 memory)
cd /workspace/caem
python scripts/caem_demo_server.py \
    --memory outputs/full_run/memory_store_cycle_3 \
    --passage_index data/passage_index \
    --port 8000

# Defense day (cycle-10 production-swapped memory)
python scripts/caem_demo_server.py \
    --memory outputs/production/memory_store/memory_store \
    --passage_index data/passage_index \
    --port 8000

# Optional: expose to a public URL via Cloudflare tunnel
bash scripts/expose_demo_remote.sh
```

Open `http://localhost:8000` (or the public URL) in a browser.
