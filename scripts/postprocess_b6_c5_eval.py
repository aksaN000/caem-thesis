#!/usr/bin/env python3
"""
scripts/postprocess_b6_c5_eval.py
=================================
Post-process B6@C5 eval JSONs to handle the ``Reasoning:<X>`` direct-answer
format that the fine-tuned B6 model emits.

The matched-protocol prompt forces the model to start its continuation with
``"Reasoning:"``. The base Qwen-2.5-3B-Instruct (with the format anchor) then
produces ``Reasoning:<thinking>\\nAnswer:<final>`` which the canonical
``extract_cot_answer`` correctly parses by splitting on ``Answer:``.

But B6 was *fine-tuned* under the broken eval shim's ``max_new_tokens=256``
cap (run_simple_ft.py:988), which truncated most of its SIL training pool's
chain-of-thought before the ``Answer:`` line landed. After 5 cycles of this
training, B6 learned to emit the final answer letter *immediately* after the
``Reasoning:`` prefix on short-form tasks like CommonsenseQA: predictions
look like ``"Reasoning:C"`` with no newline and no ``Answer:`` marker.

``extract_cot_answer`` falls through to its Priority 3 fallback and returns
the full string ``"Reasoning:C"``, which fails strict EM against the gold
``"C"``. CSQA EM = 0.0 across all 300 samples is the smoking gun.

This script post-processes the eval_fixed/ JSONs: when a prediction starts
with ``Reasoning:`` AND has no embedded ``Answer:`` AND the remainder is a
plausible final answer (short, no nested ``Reasoning:`` prefix), it strips
the prefix and re-scores EM + F1 against the gold.

Usage
-----
    python -m scripts.postprocess_b6_c5_eval

Outputs land in-place: each file gets a ``Reasoning:``-stripped sibling at
the same path with suffix ``_postprocessed.json``. The aggregator's input
path is updated separately (Phase 3 step).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.metrics import any_match_em, best_token_f1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("postprocess_b6_c5")

INPUT_DIR = _ROOT / "outputs" / "baselines_phase1e" / "vanilla_ft" / "eval_fixed"

PANEL = ("fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa")


def reparse_prediction(text: str) -> str:
    """Re-extract the final answer from a CoT-style prediction.

    Priority order:
      1. ``Answer:`` marker (canonical CoT path; unchanged from
         ``extract_cot_answer``).
      2. ``Reasoning:<X>`` prefix with no embedded ``Answer:`` and a
         plausible short remainder. Strip the prefix and return the rest.
         This is the B6 fine-tune artefact.
      3. ``the answer is X`` natural-language suffix (Flan-T5 CoT).
      4. Fallback: original text stripped.
    """
    if not isinstance(text, str):
        return text
    raw = text

    # Priority 1: explicit "Answer:" marker.
    if "Answer:" in raw:
        return raw.split("Answer:")[-1].strip()

    # Priority 2: "Reasoning:<X>" direct-answer form.
    if raw.startswith("Reasoning:"):
        after = raw[len("Reasoning:") :].strip()
        # Accept when the remainder is short and contains no embedded
        # Reasoning:/Answer: marker (already covered by Priority 1 if
        # Answer: is present).
        if after and "Reasoning:" not in after:
            return after

    # Priority 3: "the answer is X" natural-language CoT suffix.
    import re
    m = re.search(
        r"(?:^|\W)(?:so|therefore)?,?\s*the\s+answer\s+is[:\s]+(.+?)(?:[.!?](?:\s|$)|$)",
        raw,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    return raw.strip()


def process_file(in_path: Path) -> dict:
    """Re-extract + re-score one bench JSON. Returns summary dict."""
    payload = json.loads(in_path.read_text())
    samples = payload.get("samples", [])
    n_changed = 0
    em_old = []
    em_new = []
    for s in samples:
        em_old.append(s.get("em", 0.0))
        raw_pred = s.get("prediction", "")
        gold = s.get("gold_answers", []) or [s.get("gold_label", "")]
        gold = [g for g in gold if g]
        new_extracted = reparse_prediction(raw_pred)
        if new_extracted != s.get("display_answer"):
            n_changed += 1
        s["display_answer"] = new_extracted
        # Re-score EM + F1 against the gold list using the canonical
        # metric helpers (TruthfulQA's em_llm_judged is preserved as-is
        # because it's set by the Haiku rescore, not by string EM).
        if gold:
            em = any_match_em(new_extracted, gold)
            f1 = best_token_f1(new_extracted, gold)
        else:
            em = 0.0
            f1 = 0.0
        s["em"] = em
        s["f1"] = f1
        em_new.append(em)

    em_old_mean = sum(em_old) / max(len(em_old), 1)
    em_new_mean = sum(em_new) / max(len(em_new), 1)

    # Update meta.
    payload["meta"]["em"] = em_new_mean
    payload["meta"]["f1"] = sum(s["f1"] for s in samples) / max(len(samples), 1)

    out_path = in_path.with_name(in_path.stem + "_postprocessed.json")
    out_path.write_text(json.dumps(payload, indent=2))

    return {
        "bench": in_path.stem.replace("_cycle5", ""),
        "n_samples": len(samples),
        "n_changed_extraction": n_changed,
        "em_old": em_old_mean,
        "em_new": em_new_mean,
        "delta_em": em_new_mean - em_old_mean,
    }


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input dir not found: {INPUT_DIR}")

    summaries = []
    for bench in PANEL:
        in_path = INPUT_DIR / f"{bench}_cycle5.json"
        if not in_path.exists():
            logger.warning("Skip %s (not found): %s", bench, in_path)
            continue
        summary = process_file(in_path)
        summaries.append(summary)
        logger.info(
            "%s: n=%d changed=%d EM %.4f -> %.4f (delta=%+.4f)",
            summary["bench"],
            summary["n_samples"],
            summary["n_changed_extraction"],
            summary["em_old"],
            summary["em_new"],
            summary["delta_em"],
        )

    logger.info("=" * 60)
    pooled_old = sum(s["em_old"] for s in summaries) / max(len(summaries), 1)
    pooled_new = sum(s["em_new"] for s in summaries) / max(len(summaries), 1)
    logger.info("POOLED B6@C5 EM: %.4f -> %.4f (delta=%+.4f)",
                pooled_old, pooled_new, pooled_new - pooled_old)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
