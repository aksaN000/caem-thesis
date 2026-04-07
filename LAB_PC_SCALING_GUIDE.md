# CAEM Hardware Scaling Guide (Lab PC / 4090 / 5090 Edition)

**Purpose:** This document records which hyperparameters to change when moving from
the thesis-standard RTX 3060 config to a higher-capacity machine, with honest
expected gains and the academic justification (or lack thereof) for each change.

> **Thesis rule:** Only change a hyperparameter if you can defend it in viva.
> Speed-only changes (batch size, theta_prev placement) are always safe.
> Accuracy-affecting changes (passage count, sc_chains_m) require justification.

---

## Summary Table

| Parameter | Thesis config | Scaled config | Type of gain | Expected accuracy Δ | Safe to change? |
|---|---|---|---|---|---|
| `max_passages` | 500,000 | 5,000,000 | Accuracy (HotpotQA only) | +1–3% EM on HotpotQA | ✅ Yes, note in 5.3 |
| `target_episodes` | 200 | 1,000 | Accuracy (Cycle 1 only) | +2–5% EM at Cycle 1 | ✅ Yes, converges by Cycle 3 |
| `sc_chains_m` | 3 | 10 | Accuracy (marginal) | +0.5–1.5% EM | ⚠️ Requires justification |
| `se_samples_k` | 10 | 20 | Accuracy (negligible) | <1% EM | ⚠️ Not worth it |
| `mc_dropout_k` | 5 | 10 | Accuracy (negligible) | <1% | ⚠️ Not worth it |
| `batch_size` | 4 | 16–32 | Speed only | 0% | ✅ Always safe |
| `theta_prev` placement | CPU (PCIe) | GPU (native) | Speed only | 0% | ✅ Always safe |

---

## 1. Passage Index (`max_passages`)

**Thesis config:** 500,000 passages
**Scaled config:** 5,000,000 passages
**Command:**
```bash
python scripts/build_passage_index.py --max_passages 5000000 --output_dir data/passage_index
```

**Honest expected gain:**
- HotpotQA: +1–3% EM. HotpotQA requires retrieving two connected passages for multi-hop
  reasoning. A denser index increases the chance both passages are present.
- FEVER: +1–2% label accuracy. Richer claim evidence coverage.
- TruthfulQA: No meaningful gain. Questions test model knowledge, not retrieval breadth.
- StrategyQA: No meaningful gain. Boolean reasoning, not fact lookup.

**Build time:** ~30 min on 4090, ~5–6 hours on RTX 3060.
**Storage:** ~15 GB (vs ~1.5 GB for 500K).

**Thesis note (§5.3):** *"The primary experiment uses a 500K-passage Wikipedia index
consistent with Flan-T5-Large's 780M parameter capacity. A 5M-passage high-capacity
variant was evaluated on the lab GPU and reported in Appendix B."*

---

## 2. Cold-Start Seeding (`target_episodes`)

**Thesis config:** 200 episodes per benchmark
**Scaled config:** 1,000 episodes per benchmark
**Command:**
```bash
python scripts/seed_cold_start.py --target_episodes 1000 --output_dir outputs/cold_start_memory
```

**Honest expected gain:**
- Cycle 1: +2–5% EM. A denser memory at the start means more Tier 1 hits early,
  before the experiment fills memory organically.
- Cycle 3: Negligible. Both configs converge to nearly the same result because CAEM
  accumulates verified episodes throughout the experiment anyway.
- Seeding time on 4090: ~2.5 hours.

**Conclusion:** Worth doing on the lab PC since time is not a constraint there.
Not worth the 2–4 extra hours on RTX 3060.

---

## 3. Self-Consistency Chains (`sc_chains_m`)

**Thesis config:** 3 chains — [LIT] Wang et al. (2022) minimum viable consensus
**Scaled config:** 10 chains

**Honest expected gain:** +0.5–1.5% EM from cleaner `u_consistency` signal.
Diminishing returns beyond 5 chains.

**⚠️ Academic caution:** `sc_chains_m=3` is the value used in the cited paper (Wang et al. 2022).
Changing to 10 without ablation evidence opens the question: *"Why 10 and not 5 or 20?"*
If you run this, add an ablation: sc_chains ∈ {3, 5, 10} and report in Appendix B.
Do NOT change the thesis main-result config from 3.

---

## 4. Semantic Entropy Samples (`se_samples_k`)

**Thesis config:** 10 — [LIT] Farquhar et al. (2024) standard
**Scaled config:** 20

**Honest expected gain:** <1% EM. Statistically negligible.

**Verdict:** Not worth changing. `se_samples_k=10` is directly cited from the source
paper and is the most defensible value in viva. Increasing it doubles Stage 4a
inference time for no measurable thesis benefit.

---

## 5. MC Dropout (`mc_dropout_k`)

**Thesis config:** 5 passes
**Scaled config:** 10 passes

**Honest expected gain:** <1%. Minor improvement in uncertainty variance estimate.
Not worth changing for thesis purposes.

---

## 6. Training Batch Size (`batch_size`) — SPEED ONLY

**Thesis config:** 4 (RTX 3060, 12GB VRAM safe)
**Scaled config:** 16 on 4090 (24GB), 32 on 5090 (32GB)

**Accuracy impact:** Zero. Batch size does not affect final accuracy at these scales.

**Speed impact:** Fine-tuning epochs run ~4× faster at batch=16, ~6–8× faster at batch=32.
This is the single fastest change to make when moving to the lab PC — change it immediately.

```python
# caem/config.py — only this line changes on lab PC
batch_size = 16   # 4090 (24GB VRAM)
# batch_size = 32  # 5090 (32GB VRAM)
```

---

## 7. L2 Penalty: `theta_prev` on GPU — SPEED ONLY

**Current implementation (all devices):** `_l2_penalty()` computes the EWC penalty
entirely on CPU — `p.detach().cpu() - p0` — then transfers only the resulting scalar
back to GPU. This was fixed in Session 26 (the original code incorrectly called
`p0.to(self.device)` inside the batch loop, moving 3 GB of weights per batch).
The CPU implementation is correct and VRAM-efficient on all devices.

**Scaled config (4090/5090 only):** For 24GB+ VRAM GPUs, you can eliminate the
per-batch CPU↔GPU parameter transfer entirely by moving `theta_prev` to GPU **once**
before the training loop starts, then keeping it there for all batches.

**Accuracy impact:** Zero. Mathematically identical L2 penalty.

**Speed impact:** ~3–4× faster fine-tuning per cycle on 24GB+ VRAM GPUs.
This is the second most impactful change after batch size.

**Code change** (in `caem/training/self_improvement.py`, inside `_finetune()`):

Step 1 — Add one line just before the epoch loop (after `self.model.to(self.device)`):
```python
# Move theta_prev to GPU once — eliminates PCIe round-trips during training.
# Only do this if VRAM >= 24 GB.
theta_prev_gpu = [p0.to(self.device) for p0 in theta_prev]
```

Step 2 — Change the `_l2_penalty()` call inside the batch loop from:
```python
l2_loss = self._l2_penalty(theta_prev)
```
to:
```python
l2_loss = self._l2_penalty(theta_prev_gpu)
```

Step 3 — Update `_l2_penalty()` to skip the `.cpu()` detach (since both tensors
are now on GPU):
```python
def _l2_penalty(self, theta_prev: List[torch.Tensor]) -> torch.Tensor:
    """Compute ||theta - theta_prev||^2. theta_prev must already be on self.device."""
    penalty = torch.tensor(0.0, device=self.device)
    for p, p0 in zip(self.model.parameters(), theta_prev):
        diff = p - p0   # both on GPU, no PCIe cost
        penalty = penalty + (diff ** 2).sum()
    return penalty
```

**Only apply this if VRAM >= 24 GB.** On RTX 3060, the current CPU implementation
is correct and optimal — do not apply this patch.

---

## Lab PC Run Checklist

after getting university GPU access (4090 or 5090), do this in order:

1. Open `caem/config.py`:
   - Set `batch_size = 16` (4090) or `batch_size = 32` (5090)
   - Everything else stays at thesis-standard values

2. Apply the `theta_prev` GPU patch in `self_improvement.py` (speed only, no accuracy change)
   — see Section 7 above for the exact 3-step code change

3. Build the 5M passage index (optional — run in parallel/background):
   ```bash
   python scripts/build_passage_index.py --max_passages 5000000 --output_dir data/passage_index_5m
   ```

4. Run the main thesis experiment first with 500K passages (thesis-standard numbers):
   ```bash
   python scripts/run_experiment.py --n_questions 5000 --passage_index data/passage_index ...
   ```

5. After thesis numbers are confirmed, optionally run the 5M variant for Appendix B.

**Do not wait for the 5M index before running the main experiment.**
Thesis numbers come from 500K. Appendix B numbers come from 5M. Keep them separate.

---

## Expected total time on lab GPU

| Phase | 4090 | 5090 |
|---|---|---|
| Full experiment (n=5000, Cycle 0→3) | ~11–14h | ~7–10h |
| Purity validation | ~30 min | ~20 min |
| Ablations (9 configs, n=500) | ~3–5h | ~2–3h |
| Calibration | ~45 min | ~30 min |
| **Total** | **~16–20h** | **~10–14h** |
