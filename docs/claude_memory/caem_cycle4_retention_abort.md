---
name: Cycle 4 retention guard fired — first in-flight rollback, stop trajectory at C5 (2026-05-14)
description: C4 SIL aborted on TriviaQA-test probe ratio 0.8734 < 0.93 floor; weights rolled back to C3; decision to stop trajectory at C5 regardless of outcome
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Event (2026-05-14 08:56 UTC):** Cycle 4 SIL aborted on the retention guard. Worst probe TriviaQA-test dropped 0.395 → 0.345 (ratio 0.8734 < 0.93 floor). LoRA adapter saved to `outputs/full_run/cycle_4/adapter/` with `aborted=True` flag. Weights restored to cycle-3 θ_prev. Memory preserved (asymmetric rollback as designed).

**Per-probe receipts at C4:**
- MMLU: 0.61 → 0.6364 (ratio 1.0432, still gaining)
- TriviaQA-test: 0.395 → 0.345 (ratio 0.8734, FAIL)
- CommonsenseQA-test: 0.825 → 0.770 (ratio 0.9333, marginal)

**Why C4 failed where C1-C3 passed:** C4 SIL batch was 2881 episodes vs C3's 1542 (cycle-3's deferred-buffer promotions pushed up the C4 training pool). 2× gradient signal in one shot exceeded the implicit LoRA envelope.

**Thesis-side implications (POSITIVE):**
- First in-flight firing of the retention rollback — direct empirical receipt for H4 (retention floor) AND for load-bearing differentiator #4 (asymmetric rollback: memory preserved while adapter reverts).
- Ablation A3 "No retention guard" registered in `caem/ablation/variants.py` is exactly what this cycle's success exemplifies — the rollback is the mechanism removed in A3.

**Decision (locked 2026-05-14):** Let C5 attempt run. Stop trajectory regardless of C5 outcome.
- **C5 passes** → C4 was a one-off (2× batch), trajectory has 5-cycle parametric progression
- **C5 aborts** → saturation at C3 confirmed, two independent rollbacks = strong evidence for parametric ceiling claim (matches Theorem 4.x asymptote prediction)

After C5: kill runner, pivot to baselines + ablations + diagnostics + Ch5/Ch6 artefacts. Do NOT extend to C6-C10 — credit budget needs ~$30-40 for baseline panel (the single largest remaining thesis evidence gap, H2).

**Skip-retroverify-on-abort patch:** Committed `2347d60` to `scripts/run_experiment.py`. C4's in-flight retroverify wastes ~6h (runner has old code cached). C5 also runs on old code (we chose not to restart). Patch takes effect for any post-C5 runs (ablation reruns, baseline panel if they share the loop).

**Continual-learning extensions registered as Ch6 future work (NOT mid-trajectory):**
- Experience Replay from cold-start memory (most-natural fit; memory IS the replay buffer)
- EWC via λ_anc on LoRA path (already coded, currently disabled — flip is one config line)
- LR reduction for later cycles (cheapest intervention)
- Larger backbone (7B/13B) with more spare neurons per fine-tune

**How to apply:** When C5 closes (~17:00 UTC May 15 estimate), kill `tmux plan_a`, archive outputs to HF, then start the baseline panel (Task #112). The trajectory writeup in Ch5 §sec:retention should lead with the C4-abort empirical receipt for H4 and the C3 trajectory as the parametric high-water mark, with C4/C5 framed as the architecture self-bounding at the parametric ceiling.
