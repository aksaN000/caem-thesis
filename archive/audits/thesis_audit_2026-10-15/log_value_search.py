"""Search the run logs and artefacts for the thesis numbers the audit could not source.

Run from the root of your local caem-thesis checkout (read-only; changes nothing):

    python thesis_report/ref_check/log_value_search.py

Writes ~/caem_log_search.md (plain text). Upload that one file to Google Drive.

For every item it lists the files and lines (with line numbers) where the
value appears, so each thesis number can be traced to the log it came from.
A number matches when it appears as a token, also with extra trailing digits
(0.850 matches 0.8500 and 0.85012, but not 10.850 or 0.8501 for 0.851).
When an item has keywords, only lines that also contain one of them (or have
one within 3 lines) are reported, to cut noise for common values.
"""
import os
import re
import sys
from pathlib import Path

REPO = Path.cwd()
if not (REPO / "thesis_report").is_dir():
    sys.exit("Run this from the root of your caem-thesis checkout.")

ROOTS = ["outputs", "archive", "docs", "scripts", "logs", "data/calibration"]
EXTS = {".log", ".txt", ".json", ".jsonl", ".csv", ".md", ".out", ".err", ".yaml", ".yml"}
MAX_FILE = 2 * 1024 ** 3      # skip single files above 2 GB
MAX_HITS = 25                 # lines shown per item per file
MAX_FILES = 15                # files shown per item

# (id, what the thesis says, [numbers], [keywords or empty])
ITEMS = [
    # --- disputed between thesis locations ---
    ("D1", "C0 eval-fold stored set: 289 entries at 0.796 (abstract/ch3/ch5/ch6) vs 336 at 0.759 (tables)",
     ["289", "0.796", "336", "0.759"], ["store", "pi_store", "precision", "stored"]),
    ("D2", "retroactive prunes: total 1094 (ch5) vs 56/271/352/382 = 1061 (table, app C)",
     ["1094", "1061", "56", "271", "352", "382"], ["prune", "retro"]),
    ("D3", "deferred-buffer promotions: 620 (ch5) vs 650 (tables) vs 131/220 at C2/C3; local regen estimates 131/1144/1468",
     ["620", "650", "131", "220", "1144", "1468"], ["promot", "reconsider", "deferred", "buffer"]),
    ("D4", "TruthfulQA 'B1 zero-shot floor' 0.27 (ch5:473, ch6) vs B1 TQA EM 0.433 (tables)",
     ["0.27", "0.270", "0.433", "0.4333"], ["truthful", "tqa", "zero_shot", "b1"]),
    ("D5", "RUC ablation TQA EM 0.310 (on) vs 0.210 (off); off-arm aborted at C5",
     ["0.310", "0.210", "0.31", "0.21"], ["truthful", "ruc", "classifier", "ablation", "abort"]),
    ("D6", "appendix C CES column 0.557/0.567/0.547/0.548/0.556 vs tab_ces_caem_trajectory 0.678/0.704/0.703/0.690/0.710",
     ["0.557", "0.567", "0.547", "0.548", "0.556", "0.678", "0.704", "0.703", "0.690", "0.710"], ["ces"]),
    ("D7", "stream-chunk storage rate 36-43% (ch3:233) vs ~52% (tables); admissions 2443/2477/2469/2484/2339 vs 2456/2494/2490/2522/2380",
     ["2443", "2477", "2469", "2484", "2339", "2456", "2494", "2490", "2522", "2380"], ["admit", "stored", "store", "chunk"]),
    ("D8", "temperature scalar trajectory 1.089/0.836/0.740/0.869 (C1..C5) and ceiling",
     ["1.089", "0.836", "0.740", "0.869"], ["temperature", "temp", "tau", "scal"]),
    ("D9", "MMLU retention ratios 1.041/0.975/0.992/1.018/0.999 (thesis) vs 100/93.7/95.28/97.83/95.93 (experiment_summary)",
     ["1.041", "0.975", "0.992", "1.018", "0.999", "93.7", "95.28", "97.83", "95.93"], ["mmlu", "retention", "ratio"]),
    ("D10", "safety override firings (thesis prose: none of 9000; table: 1 on StrategyQA, u_pre 0.376)",
     ["0.376"], ["safety", "override", "u_pre"]),
    # --- unsourced in the thesis ---
    ("U1", "RUC training data: 3208 rows, 1846 helpful, 1362 hurt",
     ["3208", "1846", "1362"], []),
    ("U2", "RUC fit: pooled OOF AUROC 0.850, per-fold sd 0.008, drift 0.011, acc 77.2%, confusion 1554/292/923/439, recall 0.842, specificity 0.678",
     ["0.850", "0.842", "0.678", "1554", "923", "439", "77.2"], ["auroc", "auc", "recall", "spec", "ruc", "fold", "confusion"]),
    ("U3", "RUC per-bench AUROC (OBQA 0.686, TriviaQA 0.936, FEVER 0.718, CSQA 0.728, TQA 0.732, StrategyQA 0.852); probe n=11980 AUROC 0.771; p_IK coef -1.77; C=0.5",
     ["0.686", "0.936", "0.718", "0.728", "0.732", "0.852", "11980", "0.771", "-1.77"], ["auroc", "auc", "coef", "ruc", "probe"]),
    ("U4", "C0 cal-fold sweep: tau=0.65 admits 70/105, tau=0.60 306/478, pi_store ~0.64",
     ["70", "105", "306", "478", "0.64", "0.640"], ["tau", "sweep", "store", "admit", "0.65", "0.60"]),
    ("U5", "split-conformal: 0.9643 over 56 stored entries",
     ["0.9643", "0.964"], []),
    ("U6", "Qwen judge Pearson rho 0.5843 (floor 0.70)",
     ["0.5843", "0.584"], []),
    ("U7", "appendix C ECE-before: 0.2761, 0.0508, 0.0509, 0.0451, 0.0551",
     ["0.2761", "0.0508", "0.0509", "0.0451", "0.0551"], []),
    ("U8", "equilibrium fit: c_inf 0.548, a -0.018, k 0.350, R2 0.207",
     ["0.207", "-0.018", "0.350"], ["fit", "r2", "equilib", "c_inf", "saturat"]),
    ("U9", "appendix C probe anchors: TriviaQA-test 0.350/0.395; pristine 0.610/0.395/0.825",
     ["0.350", "0.395", "0.610", "0.825"], ["probe", "pristine", "test", "retention"]),
    ("U10", "appendix D: ROUGE-rule zero-shot TQA 0.803; chi2 0.232, 56.860, 76.891, 0.010",
     ["0.803", "56.86", "76.89", "0.232"], []),
    ("U11", "appendix G: ~53% stale-calibration cycle-1 prune; FEVER id 13957; 'nineteen hundred or more' fold",
     ["13957", "53"], ["prune", "stale", "13957"]),
    ("U12", "evaluation wall-time 75-95 min per cycle (app E:166); tier latency ~5.7-6.1 h per cycle",
     [], ["eval", "evaluation", "elapsed", "wall", "minutes", "took"]),
    ("U13", "CSQA train pool ~10k after dedup; FEVER 145k; TriviaQA 87k; StrategyQA dev 687",
     ["145", "87", "687"], ["dedup", "pool", "train", "split"]),
    ("U14", "C-sensitivity: cal-fold per-store precision spread <= ~1 pp across five C values",
     ["0.010", "0.025", "0.050", "0.100", "1.000"], ["sweep", "precision", "boost", "C="]),
    ("U15", "wall time ~1 s and ~7 s per consolidation; 4 sibling clusters rejected for answer mismatch",
     [], ["consolidat", "answer_mismatch", "answer-mismatch", "sibling"]),
]


def num_regex(n):
    # token boundary on the left; allow extra trailing digits after a decimal value
    left = r"(?<![\d.])"
    if "." in n:
        return re.compile(left + re.escape(n) + r"\d*(?![\d])")
    return re.compile(left + re.escape(n) + r"(?![\d.]\d|\d)")


def iter_files():
    for r in ROOTS:
        base = REPO / r
        if not base.exists():
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in {"hf_cache", "__pycache__", ".git", "node_modules"}]
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in EXTS or p.name.endswith(".log"):
                    try:
                        if p.stat().st_size <= MAX_FILE:
                            yield p
                    except OSError:
                        pass


files = list(iter_files())
print(f"scanning {len(files)} text files under {', '.join(ROOTS)} ...")
compiled = [(i, d, [num_regex(n) for n in nums], [k.lower() for k in kws]) for i, d, nums, kws in ITEMS]
hits = {i: {} for i, *_ in ITEMS}

for fp in files:
    rel = str(fp.relative_to(REPO))
    window = []  # last 3 lowercase lines for keyword proximity
    pending = []  # (item, lineno, line, deadline) waiting for a keyword in the next 3 lines
    try:
        fh = open(fp, "r", errors="replace")
    except OSError:
        continue
    with fh:
        for ln, line in enumerate(fh, 1):
            if len(line) > 4000:
                line = line[:4000]
            low = line.lower()
            # resolve pending matches that wait for a nearby keyword
            still = []
            for item, pln, pline, deadline, kws in pending:
                if any(k in low for k in kws):
                    hits[item].setdefault(rel, []).append((pln, pline))
                elif ln < deadline:
                    still.append((item, pln, pline, deadline, kws))
            pending = still
            for item, _, regs, kws in compiled:
                if regs:
                    if not any(r.search(line) for r in regs):
                        continue
                elif not any(k in low for k in kws):
                    continue
                if not kws or not regs:
                    hits[item].setdefault(rel, []).append((ln, line.rstrip()))
                elif any(k in low for k in kws) or any(any(k in w for k in kws) for w in window):
                    hits[item].setdefault(rel, []).append((ln, line.rstrip()))
                else:
                    pending.append((item, ln, line.rstrip(), ln + 4, kws))
            window = (window + [low])[-3:]

out = [f"# Log/artefact search for unsourced and disputed thesis numbers\n\nrepo: {REPO}\n"
       f"files scanned: {len(files)} (roots: {', '.join(ROOTS)})\n"]
for i, desc, nums, kws in ITEMS:
    h = hits[i]
    total = sum(len(v) for v in h.values())
    out.append(f"\n## {i}: {desc}\nsearched numbers: {nums or '-'}; keywords: {kws or '-'}; "
               f"matching lines: {total} in {len(h)} files\n")
    for rel, lines in sorted(h.items(), key=lambda kv: -len(kv[1]))[:MAX_FILES]:
        out.append(f"\n### {rel}  ({len(lines)} lines)\n")
        for ln, text in lines[:MAX_HITS]:
            out.append(f"  {ln}: {text[:300]}\n")
        if len(lines) > MAX_HITS:
            out.append(f"  ... {len(lines) - MAX_HITS} more\n")
    if len(h) > MAX_FILES:
        out.append(f"\n... {len(h) - MAX_FILES} more files\n")

dst = Path.home() / "caem_log_search.md"
dst.write_text("".join(out))
print(f"wrote {dst} ({dst.stat().st_size // 1024} KB). Upload it to Google Drive.")
