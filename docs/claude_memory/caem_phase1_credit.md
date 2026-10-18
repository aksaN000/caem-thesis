---
name: CAEM Phase 1 — current Vast credit is $102 (as of 2026-04-21)
description: Current live Vast.ai credit state for Phase 1 autonomous execution. Supersedes the older $42.15-budget memory.
type: project
originSessionId: 2f06df76-4467-4646-9e32-0a9234c27ac4
---
On 2026-04-21, user reported current Vast.ai credit balance is **$102**
on the RTX 5090 instance running CAEM Phase 1 (Branch C).

**Why:** User topped up (or the previous $42.15 baseline was low). This
puts Plan A + the previously-deferred Steps 16–18 ablation sweep
within budget where they weren't before.

**How to apply:**

- Full Plan A (Steps 5–15 + 19–20) projected cost ~$50 — leaves ~$52 cushion.
- Plan A + Steps 16–18 (ablation screening + confirmatory) projected ~$85
  total — leaves ~$17 cushion, which is tight but workable.
- The old "$20 contingent topup at Step 13 if credit < $16" rule
  (`caem_phase1_topup_contingency.md`) is **obsolete**: the $102 balance
  already clears that threshold with headroom.
- Before committing to Steps 16–18, verify free disk on the 150 GB Vast
  instance — Step 18 confirmatory checkpoints can exceed 100 GB and may
  need rolling deletion or HF-Hub offload to fit.
- Read live credit via `vastai show instances` or the Vast console before
  any big-spend decision; do not trust this memory as live state —
  overwrite it when credit changes materially.

**Storage note:** On 2026-04-21 the 150 GB instance disk was at 59 GB
free before cleanup, 63 GB free after deleting the deprecated
`flan-t5-large` HF cache (3 GB) and pip cache (916 MB). The `data/
passage_index/` directory (68 GB) is the immovable consumer. Any
Step-7/16–18 run that keeps all intermediate checkpoints will
overflow this disk; rolling-N checkpoint retention or HF-Hub offload
is required for the ablation sweep.
