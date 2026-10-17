"""Final-version evaluation sources (2026-10-17, MASTER_ISSUES M-A/M-B): one place that says which file holds each
system's answers on each benchmark, and the one correctness label every analysis uses.

Label: `em_final` (eval.metrics.score_uniform on FEVER / CommonsenseQA / StrategyQA / TriviaQA strict; the Haiku
judgement under the final-answer protocol on TruthfulQA), written into the answer files by
scripts/rescore_truthfulqa.py and scripts/rescore_uniform.py. A missing label stops the analysis.

Files:
  CAEM C0-C5          outputs/full_run/eval/<bench>_cycle<c>.json (answers and verifier fields in one file)
  Phase 1d C3         outputs/phase1d_eval/<bench>_cycle3.json (retrieval-utility classifier off)
  B1, B2, B5, B7, B8  outputs/rerun_v2/baselines/<name>/<bench>_cycle0.json (matched prompts); verifier fields in
                      the *_with_chm.json copy
  B6                  outputs/rerun_v2/baselines/vanilla_ft/eval/<bench>_cycle5.json (+ *_with_chm.json)
  B3, B4              prompts unchanged (CAEM's Tier 3 prompt): run-1 files (outputs/baselines_phase1e), except
                      TruthfulQA, re-run after the out-of-memory batch (outputs/rerun_v2/baselines)
"""
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
BENCHES = ("fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa")
TRAIN = ("fever", "triviaqa", "commonsense_qa")
CAEM_CYCLES = tuple(f"caem_c{c}" for c in range(6))
BASELINES = ("zero_shot", "cot", "rag", "cot_rag", "fiveshot_cot", "vanilla_ft", "flare", "semantic_entropy")
LABEL = {"zero_shot": "B1 zero-shot", "cot": "B2 chain-of-thought", "rag": "B3 RAG", "cot_rag": "B4 CoT + RAG",
         "fiveshot_cot": "B5 five-shot CoT", "vanilla_ft": "B6 vanilla fine-tune", "flare": "B7 FLARE",
         "semantic_entropy": "B8 semantic entropy", "phase1d_c3": "Phase 1d C3",
         **{f"caem_c{c}": f"CAEM C{c}" for c in range(6)}}
RUN1 = ROOT / "outputs/baselines_phase1e"
RERUN = ROOT / "outputs/rerun_v2/baselines"


def answer_path(system: str, bench: str) -> Path:
    if system.startswith("caem_c"):
        return ROOT / f"outputs/full_run/eval/{bench}_cycle{system[-1]}.json"
    if system == "phase1d_c3":
        return ROOT / f"outputs/phase1d_eval/{bench}_cycle3.json"
    if system == "vanilla_ft":
        return RERUN / "vanilla_ft/eval" / f"{bench}_cycle5.json"
    if system in ("rag", "cot_rag") and bench != "truthfulqa":
        return RUN1 / system / f"{bench}_cycle0.json"
    return RERUN / system / f"{bench}_cycle0.json"


def verifier_path(system: str, bench: str) -> Path:
    p = answer_path(system, bench)
    return p if system.startswith("caem_c") or system == "phase1d_c3" else p.with_name(p.stem + "_with_chm.json")


def _rows(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing (final-version source; see scripts/final_sources.py)")
    blob = json.loads(path.read_text())
    return blob["samples"] if isinstance(blob, dict) else blob


def id_to_em(system: str, bench: str) -> Dict[str, float]:
    """{question id: em_final} from the answer file."""
    out = {}
    for s in _rows(answer_path(system, bench)):
        if s.get("em_final") is None:
            raise ValueError(f"{answer_path(system, bench)}: em_final missing; run rescore_truthfulqa.py "
                             f"and rescore_uniform.py first")
        out[str(s["id"])] = float(s["em_final"])
    return out


def load(system: str, bench: str) -> List[dict]:
    """Verifier rows with `em` set to the final label (stored value kept as `em_stored`)."""
    em = id_to_em(system, bench)
    rows = _rows(verifier_path(system, bench))
    missing = [s["id"] for s in rows if str(s["id"]) not in em]
    if missing:
        raise ValueError(f"{verifier_path(system, bench)}: {len(missing)} rows without a label")
    return [dict(s, em_stored=s.get("em"), em=em[str(s["id"])]) for s in rows]
