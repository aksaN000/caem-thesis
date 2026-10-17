"""Compare the local outputs/ (the source of truth) with the Drive copy, file by file.

READ-ONLY: it uploads, deletes and overwrites nothing. Run from the root of
your local caem-thesis checkout, pointing at the Drive folder as mounted on
this machine (Google Drive for desktop, or an rclone mount):

    python thesis_report/ref_check/drive_sync_check.py "/mnt/g/My Drive/CAEM_thesis"

Writes ~/caem_drive_sync.md (text). Upload that file to Google Drive.

Mapping (local -> Drive), as used by the thesis README:
    outputs/full_run            -> 2_results_phase1e/full_run
    outputs/baselines_phase1e   -> 2_results_phase1e/phase1e_baselines
    outputs/cycle_0             -> 2_results_phase1e/cycle_0_cold_start
    outputs/tier1_routing_probe -> 2_results_phase1e/tier1_routing_probe
Files are compared by size and SHA-256 (files over 2 GB by size only).
"""
import hashlib
import os
import sys
from pathlib import Path

REPO = Path.cwd()
if not (REPO / "outputs").is_dir():
    sys.exit("Run from the root of your caem-thesis checkout (the one with outputs/).")
if len(sys.argv) != 2 or not Path(sys.argv[1]).is_dir():
    sys.exit('usage: python drive_sync_check.py "<path to the mounted CAEM_thesis folder>"')
DRIVE = Path(sys.argv[1]) / "2_results_phase1e"
MAP = [("outputs/full_run", "full_run"),
       ("outputs/baselines_phase1e", "phase1e_baselines"),
       ("outputs/cycle_0", "cycle_0_cold_start"),
       ("outputs/tier1_routing_probe", "tier1_routing_probe")]
HASH_LIMIT = 2 * 1024 ** 3


def files_under(base):
    out = {}
    if not base.is_dir():
        return out
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git"}]
        for f in files:
            p = Path(root) / f
            out[str(p.relative_to(base))] = p
    return out


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


lines = [f"# Local outputs/ vs Drive ({DRIVE})\n"]
totals = {"same": 0, "differ": 0, "local_only": 0, "drive_only": 0}
for loc_rel, drv_rel in MAP:
    loc, drv = files_under(REPO / loc_rel), files_under(DRIVE / drv_rel)
    same, differ, lonly, donly = [], [], [], []
    for rel in sorted(set(loc) | set(drv)):
        a, b = loc.get(rel), drv.get(rel)
        if a and not b:
            lonly.append((rel, a.stat().st_size))
        elif b and not a:
            donly.append((rel, b.stat().st_size))
        else:
            sa, sb = a.stat().st_size, b.stat().st_size
            if sa != sb or (sa <= HASH_LIMIT and sha(a) != sha(b)):
                differ.append((rel, sa, sb, a.stat().st_mtime, b.stat().st_mtime))
            else:
                same.append(rel)
    totals["same"] += len(same); totals["differ"] += len(differ)
    totals["local_only"] += len(lonly); totals["drive_only"] += len(donly)
    lines.append(f"\n## {loc_rel}  <->  {drv_rel}\nsame {len(same)} | differ {len(differ)} | "
                 f"local only {len(lonly)} | Drive only {len(donly)}\n")
    if differ:
        lines.append("\n### DIFFER (local is newer when local_mtime > drive_mtime)\n")
        from datetime import datetime as dt
        for rel, sa, sb, ma, mb in differ:
            lines.append(f"  {rel}  local {sa} B {dt.fromtimestamp(ma):%Y-%m-%d %H:%M}  |  "
                         f"Drive {sb} B {dt.fromtimestamp(mb):%Y-%m-%d %H:%M}\n")
    if lonly:
        lines.append("\n### LOCAL ONLY (missing on Drive)\n")
        lines += [f"  {rel}  {s} B\n" for rel, s in lonly]
    if donly:
        lines.append("\n### DRIVE ONLY (not in local outputs)\n")
        lines += [f"  {rel}  {s} B\n" for rel, s in donly]

lines.insert(1, f"\nTOTAL: same {totals['same']} | differ {totals['differ']} | "
                f"local only {totals['local_only']} | Drive only {totals['drive_only']}\n")
dst = Path.home() / "caem_drive_sync.md"
dst.write_text("".join(lines))
print(f"wrote {dst}. Upload it to Google Drive.")
print(lines[1])
