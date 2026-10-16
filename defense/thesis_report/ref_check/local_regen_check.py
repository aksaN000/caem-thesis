"""Local consistency check for the CAEM thesis, run on YOUR machine.

Run from the root of your local caem-thesis checkout (the one that has the
gitignored outputs/ folder), with the Python environment you used for the
thesis tables:

    python thesis_report/ref_check/local_regen_check.py

It does NOT modify your repo or your outputs/. It copies the repo (without
model weights / indexes) to a temporary folder, reruns every auto-table
generator there, and writes a report bundle:

    ~/caem_regen_check.zip

Upload that zip (or the folder ~/caem_regen_check/) to Google Drive and tell
Claude; it reads it from there. The bundle contains:
  state.txt            git branch/HEAD, unpushed commits, uncommitted files,
                       which gitignored artefact folders exist and their sizes
  local_changes.diff   your uncommitted edits to thesis_report/, scripts/,
                       caem/, eval/, README.md (text only)
  unpushed.diff        committed-but-unpushed changes to those paths
  regen_diff.txt       committed thesis tables vs tables regenerated from
                       your outputs/ (word diff)
  regen_log.txt        stdout/stderr of every generator
  auto_regenerated/    the regenerated tables
  artefact_index.txt   list of files under outputs/ (path, size, mtime)
  extra/               small summary artefacts the audit needs
"""
import datetime
import fnmatch
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path.cwd()
if not (REPO / "thesis_report").is_dir() or not (REPO / "scripts").is_dir():
    sys.exit("Run this from the root of your caem-thesis checkout.")
OUT = Path.home() / "caem_regen_check"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
PY = sys.executable
TEXT_PATHS = ["thesis_report", "scripts", "caem", "eval", "README.md"]
BIG = ["*.pt", "*.bin", "*.safetensors", "*.faiss", "*.index", "*.ckpt", "*.gguf",
       "adapter_model*", "optimizer*", "checkpoint-*", "*.pdf", "*.png", "*.zip", "*.tar*"]
MAX_BYTES = 600 * 1024 * 1024


def sh(cmd, cwd=REPO, timeout=None):
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True,
                       timeout=timeout, errors="replace")
    return p.returncode, p.stdout + p.stderr


def du(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


# 1. Repository state: branch, unpushed commits, uncommitted edits, gitignored artefacts.
state = [f"# local check {datetime.datetime.now().isoformat()}  repo={REPO}\n"]
for label, cmd in [
    ("branch / HEAD", "git rev-parse --abbrev-ref HEAD; git log -1 --format='%H %ad %s' --date=iso"),
    ("upstream", "git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>&1"),
    ("unpushed commits", "git log --oneline @{u}..HEAD 2>&1 | head -50"),
    ("uncommitted / untracked (not ignored)", "git status --porcelain | head -300"),
    ("ignored top-level entries", "git status --porcelain --ignored | grep '^!!' | head -200"),
]:
    state.append(f"\n## {label}\n{sh(cmd)[1]}")
state.append("\n## gitignored artefact folders (size)\n")
for d in ["outputs", "data", "hf_cache", "archive"]:
    p = REPO / d
    if p.exists():
        state.append(f"{d}: {du(p) / 1e9:.2f} GB\n")
        if d == "outputs":
            for sub in sorted(p.iterdir()):
                state.append(f"  outputs/{sub.name}: {du(sub) / 1e6:.1f} MB\n")
(OUT / "state.txt").write_text("".join(state))
(OUT / "local_changes.diff").write_text(sh("git diff HEAD -- " + " ".join(TEXT_PATHS))[1])
(OUT / "unpushed.diff").write_text(sh("git diff @{u} HEAD -- " + " ".join(TEXT_PATHS) + " 2>&1")[1])
print("repo state recorded")

# 2. Index of every artefact under outputs/ (so missing/renamed files are visible).
idx = []
for root, _, files in os.walk(REPO / "outputs"):
    for f in files:
        fp = Path(root) / f
        try:
            st = fp.stat()
            idx.append(f"{fp.relative_to(REPO)}\t{st.st_size}\t"
                       f"{datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')}\n")
        except OSError:
            pass
(OUT / "artefact_index.txt").write_text("".join(sorted(idx)))

# 3. Copy small summary artefacts the audit needs.
extra = OUT / "extra"
extra.mkdir()
for rel in ["outputs/full_run/all_cycle_results.json",
            "outputs/full_run/experiment_summary.csv",
            "outputs/full_run/pi_store_trajectory.csv",
            "outputs/full_run/tier_chm_trajectory.csv",
            "outputs/full_run/stochastic_equilibrium_trajectory.csv",
            "outputs/full_run/calibration_trajectory.csv",
            "outputs/baselines_phase1e/chm_comparison.json",
            "outputs/cycle_0/composite_calibration.json",
            "outputs/tab_sig_test_v2.csv"]:
    src = REPO / rel
    if src.is_file() and src.stat().st_size < 20 * 1024 * 1024:
        dst = extra / rel.replace("/", "__")
        shutil.copy2(src, dst)


# 4. Copy the repo to a scratch folder so the generators cannot touch your files.
def ignore(dirpath, names):
    out = []
    for n in names:
        full = os.path.join(dirpath, n)
        if n in (".git", "hf_cache", "node_modules", "__pycache__") or \
                any(fnmatch.fnmatch(n, pat) for pat in BIG):
            out.append(n)
        elif os.path.isfile(full) and os.path.getsize(full) > MAX_BYTES:
            out.append(n)
    return out


tmp = Path(tempfile.mkdtemp(prefix="caem_regen_"))
work = tmp / "caem-thesis"
print(f"copying repo (without weights/indexes) to {work} ...")
shutil.copytree(REPO, work, ignore=ignore, symlinks=False)
# a git repo in the copy lets us diff the regenerated tables against HEAD
sh(f"git -C '{work}' init -q && git -C '{work}' add -A thesis_report/figures/auto && "
   f"git -C '{work}' -c user.email=x@x -c user.name=x commit -qm base", cwd=work)

# 5. Rerun every table generator with its documented arguments.
A = "thesis_report/figures/auto"
RUNS = [
    "-m scripts.aggregate_stochastic_equilibrium --log outputs/full_run/experiment.log",  # must precede 5_2, which reads its CSV
    "-m scripts.aggregate_section_5_2",
    "-m scripts.aggregate_section_5_3",
    "-m scripts.aggregate_section_5_4",
    "-m scripts.aggregate_section_5_5",
    "-m scripts.aggregate_section_5_6",
    "-m scripts.aggregate_section_5_7",
    "-m scripts.aggregate_pi_store_trajectory --run_dir outputs/full_run --cycle_0_dir outputs/cycle_0",
    "-m scripts.aggregate_baseline_pooled --baselines_dir outputs/baselines_phase1e "
    "--caem_eval_dir outputs/full_run/eval --chm_json outputs/baselines_phase1e/chm_comparison.json",
    f"-m scripts.baseline_sig_tests_v2 --caem_cycle 5 --out_tex {A}/tab_sig_test.tex",
    f"-m scripts.baseline_sig_tests_v2 --caem_cycle 2 --out_tex {A}/tab_sig_test_c2.tex",
    "-m scripts.aggregate_tier_chm --run_dir outputs/full_run",
    "-m scripts.aggregate_calibration_trajectory --run_dir outputs/full_run --cycle_0_dir outputs/cycle_0 "
    f"--csv_out \"{tmp / 'calib_traj.csv'}\" --tex_out {A}/tab_calibration_trajectory.tex",
    "-m scripts.aggregate_three_headline_decomposition",
    "-m scripts.aggregate_memory_store_trajectory",
    "-m scripts.aggregate_per_tier_em",
    "-m scripts.aggregate_ces_per_system",
    "-m scripts.aggregate_survival_distribution",
    "-m scripts.aggregate_decision_diagnostics",
    f"-m scripts.generate_session5_artifacts --out_figures {A} --out_tables {A}",
    "-m scripts.make_tables outputs/full_run -v",
]
log = [f"# regen run {datetime.datetime.now().isoformat()}  python={PY}\n"]
for args in RUNS:
    cmd = f'"{PY}" {args}'
    try:
        rc, out = sh(cmd, cwd=work, timeout=3600)
    except subprocess.TimeoutExpired:
        rc, out = "TIMEOUT", ""
    print(f"[{rc}] {args}")
    log.append(f"\n===== [{rc}] {args}\n{out[-6000:]}")
MAKE_TABLES_OWN = {"tab_headline.tex", "tab_calibration.tex", "tab_grounding.tex", "tab_purity.tex",
                   "tab_continual.tex", "tab_cycle_progression.tex"}  # 5_2 owns tab_halluc_subtypes
for t in (work / "outputs/full_run").glob("*.tex"):
    if t.name in MAKE_TABLES_OWN and (work / A / t.name).exists():
        shutil.copy(t, work / A / t.name)
(OUT / "regen_log.txt").write_text("".join(log))

# 6. Diff regenerated tables against the committed ones; bundle everything.
(OUT / "regen_diff.txt").write_text(sh(f"git diff --word-diff=plain -U0 -- {A}", cwd=work)[1])
(OUT / "regen_stat.txt").write_text(sh(f"git diff --stat -- {A}; git status --short -- {A}", cwd=work)[1])
shutil.copytree(work / A, OUT / "auto_regenerated")
shutil.rmtree(tmp, ignore_errors=True)
zip_path = shutil.make_archive(str(OUT), "zip", OUT)
print(f"\nDone. Upload {zip_path} (or the folder {OUT}) to Google Drive and tell Claude.")
print((OUT / "regen_stat.txt").read_text())
