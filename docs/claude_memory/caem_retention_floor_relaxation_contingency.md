---
name: CAEM retention floor relaxation contingency (Phase 1e)
description: If Phase 1e retention guard fires before C3 completes, relax floor from 0.93 to 0.90. Remind user when retention probe ratios approach the floor.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Contingency plan registered 2026-05-21 by user.**

**Trigger:** If the multi-modal retention guard fires before cycle 3 completes (i.e., if any of MMLU / TriviaQA-test / CommonsenseQA-test ratio drops below 0.93 at the cycle 3 post-SIL retention probe), relax the floor to **0.90** rather than letting the rollback happen.

**Why:** User wants Phase 1e to reach at least the same cycle count as Phase 1d (C3) for apples-to-apples trajectory comparison. Phase 1d trajectory passed C3, aborted at C4. Phase 1e is on track to abort one cycle earlier due to bigger SIL pools (1474 episodes at C2 vs Phase 1d's ~100/cycle).

**How to apply:**

1. **At cycle 3 retention probe time** (projected ~12-15 UTC May 23, after cycle 3 SIL training completes), check the ratios.

2. **Current C2 baseline (just landed):**
   - MMLU: 0.975 (margin to 0.93 floor: +0.045)
   - TriviaQA-test: 0.949 (margin: +0.019) ⚠️ thinnest margin
   - CSQA-test: 0.982 (margin: +0.052)

3. **Projection if C1→C2 magnitude holds at C2→C3:**
   - MMLU: ~0.909 → below 0.93 floor ⚠️
   - TQA-test: ~0.882 → below 0.93 floor ⚠️
   - CSQA-test: ~0.958 → above floor ✓
   - **High probability of abort at C3 under current floor**

4. **Relaxation mechanism — two-step approach (user confirmed 2026-05-21):**

   **Step 1 (default): 0.93 → 0.90**
   - Config field: `retention_guard_floor` in `caem/config.py` (or similar — verify exact field name; possibly `retention_ratio_floor` or `multi_modal_retention_floor`)
   - Change from 0.93 → 0.90 before C3 retention probe fires
   - At 0.90 floor (linear projection of C1→C2 magnitude):
     - MMLU 0.909 → ABOVE floor ✓
     - TQA-test 0.882 → BELOW floor (would still abort) ⚠️
     - CSQA-test 0.958 → ABOVE floor ✓

   **Step 2 (escalation, only if 0.90 still aborts): 0.90 → 0.88**
   - Only invoke after seeing actual C3 retention values
   - User: "0.9, if .88 needed we can do that then after seeing guard values"
   - At 0.88 floor: all three probes projected above floor
   - Do not pre-emptively jump to 0.88; wait to see real C3 values first

   **Decision logic:**
   - If actual C3 retention probe shows ANY ratio below 0.93 → relax to 0.90, re-run guard check
   - If under 0.90 floor any ratio still below 0.90 → propose 0.88 as second step, await user confirmation
   - If actual C3 projection is gentler than linear (e.g., diminishing drift), 0.90 may pass cleanly without needing 0.88

5. **Side effect of relaxation:** allows more catastrophic forgetting before guard fires. Trade-off: deeper trajectory at the cost of more general-capability erosion. For thesis purposes, going to C3-C5 even with some drift is useful for the cycle-by-cycle comparison.

**Surface this memory:**
- When user asks for status check AFTER cycle 3 SIL training kicks off
- When the cycle 3 retention probe ratios are about to log
- Before the runner makes the abort/proceed decision

**Anti-trigger:** if C3 retention ratios stay above 0.93 naturally, no action needed. Just note "guard passed at C3, no relaxation triggered."

**Implementation note:** the floor change must happen in the RUNNING process (the runner reads config at startup, OR re-reads it per cycle). Need to verify whether config edit propagates without restart, or whether the user needs to halt → edit → resume. Phase 1d's abort mechanism was triggered by the running process reading its own retention guard floor, so config-edit-without-restart MAY NOT propagate. Worst case: halt runner, edit config, resume from cycle 3 SIL.

**Cycles still pending after this contingency:**
- C3 (likely needs relaxation)
- C4 (almost certainly needs relaxation under either floor)
- C5 (definitely)

So the relaxation, if applied at C3, applies for C4 and C5 too.
