# CAEM: Confidence-Aware Episodic Memory with Self-Improvement

*Confidence-Aware Episodic Memory with Self-Improvement for Hallucination Reduction in Large Language Models*

**CSE400 Final Year Thesis · BRAC University**
**Authors:** Md. Aksan Gony Alif, Maliha Binte Shamim, Md. Mahmudur Rahman
**Supervisor:** Dr. Md. Golam Rabiul Alam · **Co-supervisor:** Md. Tanzim Reza

CAEM answers knowledge questions with a three-tier router (memory, zero-shot generation, retrieval-augmented
generation) built on Qwen2.5-3B-Instruct. A multi-signal verifier scores every answer; verified answers are
stored in an episodic memory and used, once per cycle, to fine-tune the generator.

## Current status (2026-10-17)
The first full run (cycles 0–5) and the baselines are complete, and the thesis has been drafted from them.
An audit of that run found faults in the system, the baselines and the scoring. The plan is to fix them,
re-run the experiment once, and rewrite the results. **The single source for the plan and every known issue
is [`defense/thesis_report/ref_check/MASTER_ISSUES.md`](defense/thesis_report/ref_check/MASTER_ISSUES.md).**
No result numbers are quoted in this README; they live in the thesis and will change after the re-run.

## Repository map
| Path | What it is | In git? |
|---|---|---|
| `caem/` | The system: routing, memory, verifier, retrieval, self-improvement training, retrieval-utility classifier (`caem/ruc/`, with its own README), production server code | yes |
| `eval/` | Benchmarks, evaluation harness, metrics (EM, the composite hallucination metric), baselines B1–B8 | yes |
| `scripts/` | Command-line entry points: the experiment runner, cold start, calibration, baselines, re-scoring, and the analysis scripts that build thesis tables (see below) | yes |
| `tests/` | Unit and smoke tests | yes |
| `defense/thesis_report/` | **The live thesis** (LaTeX). `ref_check/` holds the drafting handoff, the clean-up tracker and the master plan | yes |
| `docs/` | Production runbook and end-to-end deployment description; `findings/` (diagnoses that feed the plan); architecture figure | yes |
| `archive/` | Superseded material, kept for provenance; see [`archive/README.md`](archive/README.md) | yes |
| `outputs/` | Run outputs (about 26 GB). Ignored by git except `outputs/analysis/`, the evidence files cited by the thesis | partly |
| `data/` | Passage index (61 GB FAISS + 7 GB passages), calibration and retention data, alias dictionary | no |
| `hf_cache/` | Local Hugging Face dataset cache | no |
| `caem book/` | Local build of the CAEM teaching book; its sources live on branch `book/caem-teaching-book` | no |

## Scripts: what to run and what not to
- **Pipeline entry points:** `run_experiment.py` (cycles), `seed_cold_start.py`, `run_calibration.py`,
  `fit_composite_calibration.py`, `build_passage_index.py`, `run_baseline.py` / `launch_baselines.sh` /
  `launch_baselines_v2.sh`, `run_simple_ft.py`, `rescore_baselines_through_verifier.py`, `rescore_truthfulqa.py`.
- **Final results (one command):** `bash scripts/make_final_results.sh` rebuilds every final number, table and figure
  of chapter 5 from the run outputs, in order: `rescore_truthfulqa.py` (only with `--judge`; Anthropic API),
  `rescore_uniform.py`, `recompute_chm_judged.py`, `paired_baseline_tests.py`, `build_ch5_evidence_tables.py`,
  `build_final_tables.py`. `scripts/final_sources.py` names the one answer file per system and benchmark.
  Baseline re-run on a rented GPU: `scripts/rental/final_version.sh` (provenance: `docs/rerun_v2_provenance.md`).
- **Other analysis for the current thesis:** `recompute_validation_gate.py`, `audit_signals_baselines.py`,
  `audit_caem_scoring.py`, `replay_early_stop_gate.py`, `rerun_cycle0_sweep.py`, `score_response_relevance.py`,
  `judge_off_topic.py`; post-defense explorations `probe_answer_confidence.py`, `probe_verifier_v2.py`.
- **Caution:** the older `aggregate_*` scripts write to the old `thesis_report/` path (not the live thesis), and the five
  `aggregate_section_5_*` scripts build tables when loaded (they now refuse to be imported).
  Do not run them; they are replaced by one results command in Phase 1a of the master plan.

## Data and artefacts outside the repository
- Run artefacts: Google Drive folder `CAEM_thesis/` (numbered layout `1_thesis` … `9_archive`).
- Passage index and pre-main snapshots: Hugging Face repository `aksaN000/caem-passage-index-21m` (private).

## Building the thesis
```
cd defense/thesis_report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Licence
See [`LICENSE`](LICENSE).
