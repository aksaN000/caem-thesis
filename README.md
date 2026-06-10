# CAEM — Confidence-Aware Episodic Memory with Self-Improvement

**CSE400 Final Year Thesis · BRAC University**
**Author:** Aksan Gony Alif

A three-tier hallucination-reduction architecture for open-domain question answering, built on Qwen-2.5-3B-Instruct. CAEM combines a verified episodic memory, confidence-aware routing with a retrieval-utility classifier, a nine-signal calibrated-probability verifier ensemble with a fixed-threshold storage gate, and a constrained per-cycle self-improvement loop under a multi-modal retention guard with asymmetric rollback.

| Resource | Link |
|---|---|
| Thesis report (325 pages, PDF) | [`thesis_report/main.pdf`](thesis_report/main.pdf) |
| Realised empirical artefacts (Google Drive) | <https://drive.google.com/drive/folders/1xErhsRDYe_qO0ZgUBn0WXK81eHc5jZCa> |
| Wikipedia passage FAISS index + pre-main snapshots (Hugging Face, private) | `aksaN000/caem-passage-index-21m` |
| Production runbook | [`docs/PRODUCTION_RUNBOOK.md`](docs/PRODUCTION_RUNBOOK.md) |
| Reproducibility recipe | [`thesis_report/appendix/appendix_e.tex`](thesis_report/appendix/appendix_e.tex) |
| Code-and-data availability | [`thesis_report/appendix/appendix_a.tex`](thesis_report/appendix/appendix_a.tex) §A.1 |

---

## Headline result

The realised programme closed at the **six-cycle horizon** (cycles C0 through C5) under the equilibrium-saturation gate, with cycle four aborted by the retention guard and weights rolled back to the cycle-three checkpoint while the memory store grew across the boundary under the asymmetric-rollback contract.

| Axis | Cycle 0 baseline | Cycle 5 close | Δ |
|---|---:|---:|---|
| Pooled exact match (5-benchmark panel) | 0.541 | 0.517 | −2.4 pp |
| Pooled composite hallucination metric | 0.135 | **0.109** | −19% relative |
| Multi-modal retention floor (worst probe) | 1.000 | 0.958 (C5); 0.886 at C4 abort | within ρ_min = 0.93 envelope |
| Realised commit fraction | — | 4/5 = 0.80 | strictly in (0, 1) |
| Memory store size | 431 (cold-start seed) | 12 175 entries | 28× growth |

The **retrieval-utility-classifier on/off ablation** at the cycle-three close lifts CommonsenseQA exact match by +14.0 pp and TruthfulQA exact match by +10.0 pp, collapses the retrieval-augmented tier share by −33.8 pp, and reduces the composite hallucination metric on CommonsenseQA by −34.4% relative.

Detailed per-benchmark trajectories, the eight-baseline panel, the pre-registered hypothesis verdicts, the theorem receipts, and the threats-to-validity disclosure are reported in Chapter 5 of the thesis.

---

## Architecture in one paragraph

A query passes through an eight-stage per-query pipeline: pre-routing confidence fusion (token-probability aggregate plus encoder-convergence ratio under per-benchmark temperature scaling), query encoding and FAISS lookup against the episodic memory, three-tier dispatch through a combined-score formula with a safety override that forces grounded retrieval when pre-routing confidence falls below the registered floor, a Stage 3b retrieval-utility classifier that intercepts the safety-override and score-formula fall-throughs to decide between zero-shot generation and retrieval-augmented generation on a per-query basis, tier execution, post-generation verification through the nine-signal verifier ensemble feeding the per-benchmark calibrated probability composite, the four-outcome decision tree (store, defer, abstain, discard) under a fixed-threshold rule, and the user-facing output. At every cycle boundary the cycle-boundary update passes run: retroactive re-verification of every stored episode under the locked verifier ensemble against the updated generator's outputs and the cycle-boundary-refit composite, deferred-buffer reconsideration with a three-way branch, memory consolidation against the cosine-similarity neighbourhood, parameter-efficient self-improvement fine-tuning through the bounded low-rank adaptation layer on the frozen backbone, and the multi-modal retention probe panel that aborts the cycle and rolls the adapter back when the worst per-probe ratio falls below the fixed floor while preserving the memory across the rollback.

---

## Locked configuration (deployed values)

| Component | Field | Value | Role |
|---|---|---:|---|
| Storage gate | `tau_store` | **0.60** | calibrated-probability cut for STORE |
| Storage gate | `tau_defer` | **0.45** | calibrated-probability cut for DEFER |
| Storage gate | `tau_retro` | 0.50 | retroactive prune floor |
| SIL admission | `tau_train` | **0.70** | strict training-pool floor (above `tau_store`) |
| Retrieval-utility classifier | `tau_RUC` | **0.375** | classifier-side cut on `p_RAG` |
| Retention guard | `rho_min` | **0.93** | worst-probe-ratio floor against pristine baseline |
| Router | `routing_lambda` | 0.70 | combined-score weight on similarity vs `u_stored` |
| Router | `tier1_combined_threshold` | 0.90 | Tier 1 dispatch floor |
| Router | `tier2_similarity_threshold` | 0.75 | Tier 2 dispatch floor |
| Router | `safety_u_pre_min` | 0.38 | safety-override floor on `u_pre` |
| Composite | shrinkage prior `alpha` | 0.6 | per-benchmark shrinkage toward pooled fit |
| Composite | boost `C` | 0.01 | L2 regulariser on Cherian boost layer |
| Composite | active signals | 9 | `u_token, u_dropout, u_internal, s_avg, p_entail, p_ground_max, p_ground_mean, p_ground_atomic, q_a_relevance` |
| Composite | retired signals | 2 | `h_norm` (registered then retired at C0); `p_contra` (structurally zero under MiniCheck) |
| Generator | backbone | Qwen-2.5-3B-Instruct | frozen across the cycle |
| Generator | LoRA | r=32, α=64, dropout=0.05 | seven linear projections, lr 2e-4 |
| Trajectory | registered cap | 10 cycles | upper bound on the SIL loop |
| Trajectory | realised closure | 6 cycles (C0 through C5) | equilibrium-saturation gate fired at the cycle-five close |

Every value is anchored in `caem/config.py`; the registry is mirrored in Appendix A of the thesis.

---

## Repository layout

```
caem/                         core package (pipeline, memory, routing, verification, training)
scripts/                      cycle runner, calibration scripts, baseline drivers, aggregators
eval/                         shared per-benchmark harness (EM, em_llm_judged, CHM, CES)
tests/                        pytest suite (storage gate, deferred buffer, retention, SIL, verifier)
docs/                         production runbook, demo quickstart, panel-demo script
thesis_report/                LaTeX thesis source (Ch1-Ch6 + 7 appendices + auto-generated tables)
archive/                      historical design diaries and earlier-phase runners
data/                         passage index, calibration pairs, alias dictionary (gitignored)
hf_cache/                     Hugging Face cache (gitignored)
outputs/                      per-cycle artefacts and logs (gitignored; mirrored on Google Drive)
```

A detailed module map with the role of each file is in Appendix A of the thesis (`thesis_report/appendix/appendix_a.tex`).

---

## Pre-registered hypotheses (verdicts)

The thesis registers five falsifiable hypotheses ahead of the empirical chapters; the realised cycle-five verdicts are:

| ID | Hypothesis | Verdict |
|---|---|---|
| H1 | Verifier balanced accuracy clears one-half precondition every cycle (`thm:purity` precondition) | **supported** |
| H2 | Pool purity non-decreasing under the precondition (`thm:purity`) | **supported** (band reading) |
| H3 | Late-cycle pool-purity increments enter a bounded stationary band (`thm:convergence`) | **supported** within horizon |
| H4 | Composite hallucination at the realised horizon approaches a parameter-bounded asymptote (`thm:asymptotic-elim`) | *scope-limited* (content-hash-disjoint eval contract pins Tier 1 rate) |
| H5 | Per-cycle Tier 1 hit rate grows monotonically (operational claim) | *scope-limited* (same disjointness contract; per-query cost reduction not measurable in six cycles) |

The full theorem-by-theorem proof-and-receipt adjudication is reported in Chapter 5 §5.5.1 and §5.5.2 of the thesis.

---

## Reproducibility

Every reported number is regenerable from the realised seed plus the published artefact bundle. The end-to-end recovery sequence is:

1. Clone this repository.
2. Pull the realised artefacts from the Google Drive folder at <https://drive.google.com/drive/folders/1xErhsRDYe_qO0ZgUBn0WXK81eHc5jZCa> into `outputs/`.
3. Pull the Wikipedia FAISS passage index from the Hugging Face dataset host (`aksaN000/caem-passage-index-21m`; access on request) into `data/`.
4. Pin the code to commit `c5b8066` (pre-Step-7 anchor) or `700b9e3` (retrieval-utility-classifier-era anchor) depending on the trajectory phase to reproduce.
5. Run the verification checklist in Appendix E of the thesis (`thesis_report/appendix/appendix_e.tex` §E.4) to confirm every headline number matches within the registered tolerance.

The full step-by-step reproducibility recipe, including hardware-envelope notes, framework version pins, the data-preparation pipeline, and the gate-verdict reference, is Appendix E.

---

## Citation

If you reference CAEM or build on this work, please cite:

```bibtex
@misc{alif2026caem,
  author = {Aksan Gony Alif},
  title  = {Confidence-Aware Episodic Memory with Self-Improvement: An Architectural Approach to Hallucination Reduction in Open-Domain Question Answering},
  year   = {2026},
  note   = {CSE400 Undergraduate Thesis, BRAC University},
  url    = {https://github.com/aksaN000/caem-thesis}
}
```

---

## Hardware and licence

The realised trajectory ran on a single rented RTX 5090 (32 GB) on Vast.ai at approximately 0.70 USD per on-demand GPU-hour, consuming approximately 294 GPU-hours and 163 USD spread across roughly 40 calendar days of elapsed time. Detailed envelope, version pins, and the downgraded sixteen-gigabyte reproduction profile are in Appendix E.

Released under the [MIT licence](LICENSE).
