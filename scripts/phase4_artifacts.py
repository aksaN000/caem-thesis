#!/usr/bin/env python
"""Phase 4 — empirical-evidence artifact generator.

Reads whatever Phase 3 outputs are available in
``outputs/cycle_0/eval/*_cycle0.json``, ``composite_calibration.json``,
``conformal_gate.json``, and emits the LaTeX tables + matplotlib figures
the thesis cites.

Graceful degradation: if a required file is missing, the corresponding
artifact is written with a ``[PENDING — Phase 3 not yet complete]``
placeholder so the build doesn't fail. Re-run after each new JSON
lands to refresh the artifacts.

Outputs (all under ``outputs/phase4/``):

  cohen_d_table.tex            7-benchmark × 11-signal Cohen's d table (Ch5)
  composite_weight_table.tex   Per-signal weight + Cohen's d justification (Ch4)
  precision_cliff.pdf          Precision-recall Pareto frontier (Ch5)
  precision_cliff_data.tex     Companion table for the Pareto figure
  atomic_scope.pdf             Atomic-fallback rate per benchmark (Ch4)
  prompt_compliance.tex        Empty-Answer / IDK rate before vs after (Ch5)
  cross_benchmark_summary.tex  Top-line EM / STORE / precision per benchmark (Ch5)
  manifest.json                Which artifacts are filled vs PENDING

Usage:
  python scripts/phase4_artifacts.py
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("phase4_artifacts")

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "outputs" / "cycle_0" / "eval"
CYCLE0_DIR = REPO_ROOT / "outputs" / "cycle_0"
OUT_DIR = REPO_ROOT / "outputs" / "phase4"

# v2 Fix 9b: read benchmark roster from caem.config so phase-4 reports
# track the live panel. v1 hardcoded to 7-bench panel (FEVER + TriviaQA +
# NQ training; TruthfulQA + StrategyQA + ARC + ASQA transfer). v2 uses
# (FEVER + TriviaQA + HotpotQA + CSQA training; TruthfulQA + StrategyQA +
# NQ transfer). Display names extended for the new v2 benchmarks.
from caem.config import (
    TRAINING_BENCHMARKS as _CFG_TRAINING_BENCHMARKS,
    TRANSFER_BENCHMARKS as _CFG_TRANSFER_BENCHMARKS,
)
BENCHES = list(_CFG_TRAINING_BENCHMARKS) + list(_CFG_TRANSFER_BENCHMARKS)
BENCH_DISPLAY = {
    "fever": "FEVER", "triviaqa": "TriviaQA", "natural_questions": "NQ",
    "truthfulqa": "TruthfulQA", "strategyqa": "StrategyQA",
    "arc_challenge": "ARC-C", "asqa": "ASQA",
    # v2 additions
    "hotpotqa": "HotpotQA",
    "commonsense_qa": "CSQA",
}
TRAIN_BENCHES = set(_CFG_TRAINING_BENCHMARKS)
TRANSFER_BENCHES = set(_CFG_TRANSFER_BENCHMARKS)
SIGNAL_DISPLAY = [
    ("u_stored",        r"$u_{\text{stored}}$"),
    ("p_ground_mean",   r"$p^{\text{mean}}_{\text{ground}}$"),
    ("p_ground_max",    r"$p^{\text{max}}_{\text{ground}}$"),
    ("p_ground_atomic", r"$p^{\text{atomic}}_{\text{ground}}$"),
    ("p_entail",        r"$p_{\text{entail}}$"),
    ("q_a_relevance",   r"$q_{a,\text{rel}}$"),
    ("u_internal",      r"$u_{\text{internal}}$"),
    ("u_token",         r"$u_{\text{token}}$"),
    ("u_dropout",       r"$u_{\text{dropout}}$"),
    ("s_avg",           r"$s_{\text{avg}}$"),
    ("h_norm",          r"$h_{\text{norm}}$"),
]


# --------------------------------------------------------------------- #
# small helpers                                                          #
# --------------------------------------------------------------------- #


def cohen_d(positives: Sequence[float], negatives: Sequence[float]) -> float:
    if len(positives) < 2 or len(negatives) < 2:
        return float("nan")
    import statistics
    s1, s0 = statistics.stdev(positives), statistics.stdev(negatives)
    pooled = math.sqrt(((len(positives) - 1) * s1 * s1 + (len(negatives) - 1) * s0 * s0)
                       / (len(positives) + len(negatives) - 2))
    if pooled <= 0:
        return float("nan")
    return (statistics.mean(positives) - statistics.mean(negatives)) / pooled


def precision_cliff(scores: Sequence[float], labels: Sequence[int],
                    target: float, min_n: int = 20) -> Optional[Tuple[int, float, float]]:
    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    correct = 0
    last: Optional[Tuple[int, float, float]] = None
    for k, (sc, em) in enumerate(pairs, start=1):
        correct += em
        if k >= min_n:
            if (correct / k) >= target:
                last = (k, sc, correct / k)
            else:
                break
    return last


def load_eval(bench: str) -> Optional[List[Dict[str, Any]]]:
    p = EVAL_DIR / f"{bench}_cycle0.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return d.get("samples") or d.get("results") or []


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt_pending(reason: str) -> str:
    return f"% [PENDING — {reason}]\n"


# --------------------------------------------------------------------- #
# Artifact 1: Per-benchmark Cohen's d table (Ch5)                       #
# --------------------------------------------------------------------- #


def emit_cohen_d_table() -> Tuple[Path, bool]:
    """7×11 Cohen's d table, with explicit ID vs Transfer split.

    Per Ch3 §Data, training benchmarks (FEVER, TriviaQA, NQ) and transfer
    benchmarks (TruthfulQA, StrategyQA, ARC, ASQA) play distinct roles:
    the training pool is calibrated against the ID distribution, while
    transfer benchmarks probe out-of-distribution verifier generalization.
    Theorem T1's per-cycle purity guarantee uses the ID-pooled alpha
    only; transfer alpha is reported separately as a generalization
    diagnostic, not as part of the purity audit.
    """
    rows: Dict[str, Dict[str, float]] = {}
    n_present = 0
    for b in BENCHES:
        samples = load_eval(b)
        if samples is None:
            continue
        n_present += 1
        pos = [s for s in samples if s.get("em") in (1, 1.0)]
        neg = [s for s in samples if s.get("em") in (0, 0.0)]
        rows[b] = {}
        for sig, _ in SIGNAL_DISPLAY:
            p_vals = [s[sig] for s in pos if isinstance(s.get(sig), (int, float))]
            n_vals = [s[sig] for s in neg if isinstance(s.get(sig), (int, float))]
            rows[b][sig] = cohen_d(p_vals, n_vals)

    out = OUT_DIR / "cohen_d_table.tex"
    if n_present == 0:
        out.write_text(fmt_pending("no eval JSONs present yet"))
        return out, False

    # Compute ID-pooled and Transfer-pooled per-signal Cohen's d
    id_pooled: Dict[str, float] = {}
    tr_pooled: Dict[str, float] = {}
    for sig, _ in SIGNAL_DISPLAY:
        id_pos: List[float] = []
        id_neg: List[float] = []
        tr_pos: List[float] = []
        tr_neg: List[float] = []
        for b in BENCHES:
            samples = load_eval(b)
            if samples is None:
                continue
            target_pos = id_pos if b in TRAIN_BENCHES else tr_pos
            target_neg = id_neg if b in TRAIN_BENCHES else tr_neg
            for s in samples:
                v = s.get(sig)
                em = s.get("em")
                if isinstance(v, (int, float)):
                    if em in (1, 1.0):
                        target_pos.append(float(v))
                    elif em in (0, 0.0):
                        target_neg.append(float(v))
        id_pooled[sig] = cohen_d(id_pos, id_neg)
        tr_pooled[sig] = cohen_d(tr_pos, tr_neg)

    # Split the column ordering: ID benches first, then Transfer
    ordered_benches = [b for b in BENCHES if b in TRAIN_BENCHES] + \
                      [b for b in BENCHES if b in TRANSFER_BENCHES]

    lines = [
        "% Auto-generated by scripts/phase4_artifacts.py",
        "% Benchmarks present: " + ", ".join(b for b in ordered_benches if b in rows),
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Per-signal Cohen's $d$ on the post-Phase-2 Cycle-0 eval fold "
        r"(em=1 vs em=0). The training-pooled column governs Theorem T1's "
        r"purity guarantee for the SIL training pool; the transfer-pooled "
        r"column reports out-of-distribution verifier generalization to "
        r"benchmarks excluded from calibration. Positive values indicate the "
        r"signal is higher on em-correct samples; the cal-prob composite "
        r"auto-handles sign-flip signals via per-signal isotonic direction "
        r"detection.}",
        r"\label{tab:cohen-d-per-bench}",
        r"\begin{tabular}{l|" + "r" * 3 + "r|" + "r" * 4 + "r|}",
        r"\toprule",
        r"signal & \multicolumn{4}{c|}{\textbf{Training (ID, in calibration)}} "
        r"& \multicolumn{5}{c|}{\textbf{Transfer (OOD, eval only)}} \\",
        r" & " + " & ".join(BENCH_DISPLAY[b] for b in TRAIN_BENCHES if b in BENCHES) +
        r" & \textbf{ID pool} & " +
        " & ".join(BENCH_DISPLAY[b] for b in TRANSFER_BENCHES if b in BENCHES) +
        r" & \textbf{Trans pool} \\",
        r"\midrule",
    ]
    train_list = [b for b in BENCHES if b in TRAIN_BENCHES]
    transfer_list = [b for b in BENCHES if b in TRANSFER_BENCHES]
    for sig, sig_disp in SIGNAL_DISPLAY:
        cells = [sig_disp]
        for b in train_list:
            v = rows.get(b, {}).get(sig, float("nan"))
            cells.append("--" if math.isnan(v) else f"{v:+.2f}")
        v = id_pooled.get(sig, float("nan"))
        cells.append("--" if math.isnan(v) else f"\\textbf{{{v:+.2f}}}")
        for b in transfer_list:
            v = rows.get(b, {}).get(sig, float("nan"))
            cells.append("--" if math.isnan(v) else f"{v:+.2f}")
        v = tr_pooled.get(sig, float("nan"))
        cells.append("--" if math.isnan(v) else f"\\textbf{{{v:+.2f}}}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out.write_text("\n".join(lines) + "\n")
    return out, True


# --------------------------------------------------------------------- #
# Artifact 2: Composite-weight justification table (Ch4)                #
# --------------------------------------------------------------------- #


def emit_composite_weight_table() -> Tuple[Path, bool]:
    """Per-signal Pearson + Cherian boost weight + role in composite."""
    cal_path = CYCLE0_DIR / "composite_calibration.json"
    cal = load_json(cal_path)
    out = OUT_DIR / "composite_weight_table.tex"

    if cal is None:
        out.write_text(fmt_pending("composite_calibration.json not yet fitted"))
        return out, False

    calibrations = {c["signal"]: c for c in cal.get("calibrations", [])}
    boost_w = cal.get("boost_weights")
    boost_intercept = cal.get("boost_intercept", 0.0)

    lines = [
        "% Auto-generated by scripts/phase4_artifacts.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{CalProbComposite per-signal calibration on the Cycle-0 "
        r"calibration fold ($n=" + str(cal.get("metadata", {}).get("fitted_n", "")) + r"$, "
        r"em rate $" + f"{cal.get('metadata', {}).get('em_rate', float('nan')):.3f}" + r"$). "
        r"Each signal is mapped to a calibrated probability via isotonic "
        r"regression; Pearson direction is auto-detected. The Cherian "
        r"boost layer (logistic regression on calibrated log-odds) "
        r"learns per-signal weights that correct cross-signal correlations.}",
        r"\label{tab:composite-calibration}",
        r"\begin{tabular}{lrlr}",
        r"\toprule",
        r"signal & Pearson $\rho$ & direction & boost weight \\",
        r"\midrule",
    ]
    for sig, sig_disp in SIGNAL_DISPLAY:
        if sig == "u_stored":  # u_stored is the OUTPUT, not an input signal
            continue
        c = calibrations.get(sig)
        if c is None:
            row = (sig_disp, "--", "--", "--")
        else:
            pe = c.get("pearson", float("nan"))
            dr = "increasing" if c.get("increasing", True) else "decreasing"
            bw = boost_w.get(sig, float("nan")) if boost_w else float("nan")
            row = (
                sig_disp,
                f"{pe:+.3f}",
                dr,
                "--" if math.isnan(bw) else f"{bw:+.3f}",
            )
        lines.append(" & ".join(row) + r" \\")
    if boost_w is not None:
        lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{3}}{{r}}{{boost intercept}} & {boost_intercept:+.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines) + "\n")
    return out, True


# --------------------------------------------------------------------- #
# Artifact 3: Precision-cliff Pareto figure + companion table (Ch5)     #
# --------------------------------------------------------------------- #


def emit_precision_cliff() -> Tuple[List[Path], bool]:
    out_pdf = OUT_DIR / "precision_cliff.pdf"
    out_tex = OUT_DIR / "precision_cliff_data.tex"

    # Pool ID-only samples
    id_samples: List[Dict[str, Any]] = []
    transfer_samples: List[Dict[str, Any]] = []
    for b in BENCHES:
        s = load_eval(b)
        if s is None:
            continue
        if b in TRAIN_BENCHES:
            id_samples.extend(s)
        else:
            transfer_samples.extend(s)

    if not id_samples and not transfer_samples:
        out_tex.write_text(fmt_pending("no eval JSONs present yet"))
        out_pdf.write_text(fmt_pending("no eval JSONs present yet"))
        return [out_tex, out_pdf], False

    # Build precision-cliff curves
    def curve(samples: List[Dict[str, Any]]) -> Tuple[List[float], List[float], int]:
        if not samples:
            return [], [], 0
        pairs = sorted(
            [(float(s.get("u_stored", 0.0)), int(s.get("em", 0))) for s in samples],
            key=lambda p: -p[0],
        )
        precs, recalls = [], []
        correct = 0
        for k, (_, em) in enumerate(pairs, start=1):
            correct += em
            precs.append(correct / k)
            recalls.append(k / len(pairs))
        return precs, recalls, len(pairs)

    id_p, id_r, id_n = curve(id_samples)
    tr_p, tr_r, tr_n = curve(transfer_samples)

    # Companion table
    lines_tex = [
        "% Auto-generated by scripts/phase4_artifacts.py",
        f"% ID samples: n={id_n}; Transfer samples: n={tr_n}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Storage precision at fixed targets, in-distribution vs "
        r"transfer. Lower storage rate $=$ higher operating-point threshold.}",
        r"\label{tab:precision-cliff}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"target precision & ID storage rate & transfer storage rate \\",
        r"\midrule",
    ]
    for tgt in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
        f_id = precision_cliff([float(s.get("u_stored", 0.0)) for s in id_samples],
                                 [int(s.get("em", 0)) for s in id_samples], tgt)
        f_tr = precision_cliff([float(s.get("u_stored", 0.0)) for s in transfer_samples],
                                 [int(s.get("em", 0)) for s in transfer_samples], tgt)
        cell_id = f"{f_id[0] / max(1, id_n):.1%}" if f_id and id_n else "unreach"
        cell_tr = f"{f_tr[0] / max(1, tr_n):.1%}" if f_tr and tr_n else "unreach"
        lines_tex.append(f"$\\geq {int(tgt * 100)}\\%$ & {cell_id} & {cell_tr} \\\\")
    lines_tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_tex.write_text("\n".join(lines_tex) + "\n")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        if id_n > 0:
            ax.plot(id_r, id_p, label=f"ID (n={id_n})", linewidth=2)
        if tr_n > 0:
            ax.plot(tr_r, tr_p, label=f"Transfer (n={tr_n})", linewidth=2, linestyle="--")
        ax.axhline(0.80, linestyle=":", alpha=0.6, color="grey")
        ax.text(0.02, 0.81, r"$\alpha=0.20$ target", fontsize=8, color="grey")
        ax.axhline(0.70, linestyle=":", alpha=0.4, color="grey")
        ax.set_xlabel("storage rate (recall)")
        ax.set_ylabel("realised precision")
        ax.set_title("CAEM storage precision vs storage rate")
        ax.set_xlim(0, max(0.5, max(id_r + tr_r) if (id_r or tr_r) else 0.5))
        ax.set_ylim(0.4, 1.02)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_pdf)
        plt.close(fig)
    except Exception as exc:
        logger.warning("matplotlib failed (%s) — writing text placeholder for PDF", exc)
        out_pdf.write_text(fmt_pending(f"matplotlib error: {exc}"))

    return [out_tex, out_pdf], True


# --------------------------------------------------------------------- #
# Artifact 4: Atomic-decomp scope figure (Ch4)                          #
# --------------------------------------------------------------------- #


def emit_atomic_scope() -> Tuple[Path, bool]:
    out = OUT_DIR / "atomic_scope.tex"
    rows: List[Tuple[str, int, int, float, float]] = []
    any_present = False
    for b in BENCHES:
        s = load_eval(b)
        if s is None:
            continue
        any_present = True
        n = len(s)
        if n == 0:
            continue
        # Atomic fallback: p_ground_atomic exactly equals p_ground_mean
        fallback = sum(
            1 for x in s
            if isinstance(x.get("p_ground_atomic"), (int, float))
            and isinstance(x.get("p_ground_mean"), (int, float))
            and abs(x["p_ground_atomic"] - x["p_ground_mean"]) < 1e-9
        )
        # Display answer length distribution
        long_count = sum(1 for x in s if len(str(x.get("display_answer") or "").split()) >= 12)
        rows.append((b, n, fallback, fallback / n, long_count / n))

    if not any_present:
        out.write_text(fmt_pending("no eval JSONs present yet"))
        return out, False

    lines = [
        "% Auto-generated by scripts/phase4_artifacts.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Atomic decomposition scope per benchmark. Atomic falls back "
        r"to $p^{\text{mean}}_{\text{ground}}$ on short answers (length-gate "
        r"at 12 tokens, Phase 2.3); only long-form answers get genuine "
        r"per-fact NLI. Aligns with FActScore's documented scope "
        r"(Min et al.\ EMNLP 2023, §3).}",
        r"\label{tab:atomic-scope}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"benchmark & n & fallback (count) & fallback rate & long-form rate \\",
        r"\midrule",
    ]
    for b, n, fb, fb_rate, lf_rate in rows:
        lines.append(
            f"{BENCH_DISPLAY[b]} & {n} & {fb} & {fb_rate * 100:.1f}\\% & "
            f"{lf_rate * 100:.1f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines) + "\n")
    return out, True


# --------------------------------------------------------------------- #
# Artifact 5: Prompt-compliance recovery (Ch5)                          #
# --------------------------------------------------------------------- #


def emit_prompt_compliance_table() -> Tuple[Path, bool]:
    """Compares pre-Phase-2 archive vs current Phase 3 empty/IDK rates."""
    out = OUT_DIR / "prompt_compliance.tex"
    pre_dir = REPO_ROOT / "outputs" / "archive" / "pre_a2v3_broken_composite_2026-04-25" / "cycle_0" / "eval"

    def rates(d: Path) -> Dict[str, Tuple[int, float, float]]:
        out: Dict[str, Tuple[int, float, float]] = {}
        for b in BENCHES:
            f = d / f"{b}_cycle0.json"
            if not f.exists():
                continue
            with open(f) as fh:
                doc = json.load(fh)
            ss = doc.get("samples") or []
            if not ss:
                continue
            empty = sum(1 for x in ss if (x.get("display_answer") or "") == "")
            idk = sum(1 for x in ss
                       if str(x.get("display_answer") or "").strip().lower() in {
                           "i do not know.", "i don't know.", "i don't know", "not enough info"
                       })
            out[b] = (len(ss), empty / len(ss), idk / len(ss))
        return out

    pre = rates(pre_dir)
    post = rates(EVAL_DIR)

    if not pre and not post:
        out.write_text(fmt_pending("no eval JSONs present (pre or post)"))
        return out, False

    lines = [
        "% Auto-generated by scripts/phase4_artifacts.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Empty-Answer and refusal rates per benchmark, before vs "
        r"after the Phase 2 prompt-compliance + over-abstention fixes "
        r"(Phase 2.6, 2.7).}",
        r"\label{tab:prompt-compliance}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"benchmark & pre empty & post empty & pre IDK & post IDK \\",
        r"\midrule",
    ]
    for b in BENCHES:
        pre_t = pre.get(b)
        post_t = post.get(b)
        cell_pre_e = f"{pre_t[1] * 100:.1f}\\%" if pre_t else "--"
        cell_post_e = f"{post_t[1] * 100:.1f}\\%" if post_t else "--"
        cell_pre_i = f"{pre_t[2] * 100:.1f}\\%" if pre_t else "--"
        cell_post_i = f"{post_t[2] * 100:.1f}\\%" if post_t else "--"
        lines.append(
            f"{BENCH_DISPLAY[b]} & {cell_pre_e} & {cell_post_e} & "
            f"{cell_pre_i} & {cell_post_i} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines) + "\n")
    return out, True


# --------------------------------------------------------------------- #
# Artifact 6: Cross-benchmark summary table (Ch5)                       #
# --------------------------------------------------------------------- #


def emit_summary_table() -> Tuple[Path, bool]:
    out = OUT_DIR / "cross_benchmark_summary.tex"
    rows = []
    any_present = False
    for b in BENCHES:
        s = load_eval(b)
        if s is None:
            continue
        any_present = True
        n = len(s)
        em_n = sum(1 for x in s if x.get("em") in (1, 1.0))
        store = [x for x in s if x.get("decision") == "STORE"]
        store_n = len(store)
        store_em = sum(1 for x in store if x.get("em") in (1, 1.0))
        prec = store_em / store_n if store_n else float("nan")
        rows.append((b, n, em_n / n, store_n, store_n / n, prec))

    if not any_present:
        out.write_text(fmt_pending("no eval JSONs present yet"))
        return out, False

    lines = [
        "% Auto-generated by scripts/phase4_artifacts.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Cycle-0 top-line per benchmark under the Phase 2 "
        r"comprehensive patch. ID benchmarks (FEVER, TriviaQA, NQ) feed "
        r"the SIL training pool; transfer benchmarks are evaluated only.}",
        r"\label{tab:cross-bench-summary}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"benchmark & n & EM & STORE n & store rate & STORE precision \\",
        r"\midrule",
    ]
    for b, n, em_rate, store_n, store_rate, prec in rows:
        rs = "ID" if b in TRAIN_BENCHES else "T"
        lines.append(
            f"{BENCH_DISPLAY[b]} ({rs}) & {n} & {em_rate:.3f} & {store_n} & "
            f"{store_rate * 100:.1f}\\% & "
            f"{'--' if math.isnan(prec) else f'{prec:.3f}'} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines) + "\n")
    return out, True


# --------------------------------------------------------------------- #
# Main                                                                   #
# --------------------------------------------------------------------- #


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"artifacts": {}, "phase3_eval_jsons_present": []}

    for b in BENCHES:
        if (EVAL_DIR / f"{b}_cycle0.json").exists():
            manifest["phase3_eval_jsons_present"].append(b)

    print(f"\n=== Phase 4 artifact generation ===")
    print(f"Eval JSONs present: {manifest['phase3_eval_jsons_present']}")
    print(f"composite_calibration.json: {(CYCLE0_DIR / 'composite_calibration.json').exists()}")
    print(f"conformal_gate.json: {(CYCLE0_DIR / 'conformal_gate.json').exists()}\n")

    out_p, ok = emit_cohen_d_table()
    manifest["artifacts"]["cohen_d_table"] = {"path": str(out_p), "filled": ok}
    print(f"  {'[FILLED]' if ok else '[PENDING]':<10} {out_p.name}")

    out_p, ok = emit_composite_weight_table()
    manifest["artifacts"]["composite_weight_table"] = {"path": str(out_p), "filled": ok}
    print(f"  {'[FILLED]' if ok else '[PENDING]':<10} {out_p.name}")

    paths, ok = emit_precision_cliff()
    manifest["artifacts"]["precision_cliff"] = {"paths": [str(p) for p in paths], "filled": ok}
    print(f"  {'[FILLED]' if ok else '[PENDING]':<10} precision_cliff.{{tex,pdf}}")

    out_p, ok = emit_atomic_scope()
    manifest["artifacts"]["atomic_scope"] = {"path": str(out_p), "filled": ok}
    print(f"  {'[FILLED]' if ok else '[PENDING]':<10} {out_p.name}")

    out_p, ok = emit_prompt_compliance_table()
    manifest["artifacts"]["prompt_compliance"] = {"path": str(out_p), "filled": ok}
    print(f"  {'[FILLED]' if ok else '[PENDING]':<10} {out_p.name}")

    out_p, ok = emit_summary_table()
    manifest["artifacts"]["cross_benchmark_summary"] = {"path": str(out_p), "filled": ok}
    print(f"  {'[FILLED]' if ok else '[PENDING]':<10} {out_p.name}")

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  manifest -> {OUT_DIR / 'manifest.json'}")

    n_filled = sum(1 for v in manifest["artifacts"].values() if v.get("filled"))
    n_total = len(manifest["artifacts"])
    print(f"\n=== {n_filled}/{n_total} artifacts filled. Re-run after each new Phase 3 JSON. ===\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
