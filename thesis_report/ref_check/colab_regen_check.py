# CAEM thesis: regenerate every auto table from the Drive artefacts and diff it
# against the committed thesis tables. Paste this whole file into ONE Colab cell
# and run it. It never writes to your Drive artefacts: it copies them to the
# Colab disk first, and the only thing it writes to Drive is the result folder
# MyDrive/caem_regen_check/.
import os, subprocess, shutil, glob, getpass, datetime, sys

from google.colab import drive
drive.mount('/content/drive')

REPO = '/content/caem-thesis'
BRANCH = 'claude/elegant-planck-strk55'
RESULT = '/content/drive/MyDrive/caem_regen_check'

def sh(cmd, cwd=None, timeout=None):
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr

# 1. Clone the thesis repo (private: needs a GitHub token with read access).
if not os.path.isdir(REPO):
    tok = getpass.getpass('GitHub token (read access to aksaN000/caem-thesis): ').strip()
    rc, out = sh(f'git clone -q -b {BRANCH} https://{tok}@github.com/aksaN000/caem-thesis.git {REPO}')
    if rc:
        sys.exit('clone failed:\n' + out.replace(tok, '***'))
    sh(f'git -C {REPO} remote set-url origin https://github.com/aksaN000/caem-thesis.git')

# 2. Locate the canonical artefacts: MyDrive/caem-phase1a/v2_1_phase1e_main
#    (the folder linked from the GitHub README; it has the post-Vast updates).
cands = ['/content/drive/MyDrive/caem-phase1a'] + glob.glob('/content/drive/*/caem-phase1a') + \
        glob.glob('/content/drive/MyDrive/*/caem-phase1a')
ROOT = next((c for c in cands if os.path.isdir(c)), None)
if ROOT is None:
    sys.exit('Could not find the caem-phase1a folder on Drive; set ROOT by hand.')
MAIN = f'{ROOT}/v2_1_phase1e_main'
print('artefacts:', MAIN)

# 3. Copy them to local disk under the paths the scripts expect
#    (skip model weights / indexes; the table scripts don't read them).
#    Drive folder                       -> repo path
MAP = [('full_run',            'outputs/full_run'),
       ('phase1e_baselines',   'outputs/baselines_phase1e'),
       ('phase1e_baselines',   'outputs/baselines'),
       ('cycle_0_cold_start',  'outputs/cycle_0'),
       ('tier1_routing_probe', 'outputs/tier1_routing_probe'),
       ('logs',                'outputs/logs')]
excl = ' '.join(f"--exclude='{p}'" for p in
                ['*.pt', '*.bin', '*.safetensors', '*.faiss', '*.index', '*.ckpt',
                 '*.gguf', 'adapter_model*', 'checkpoint-*', 'hf_cache'])
for src, dst in MAP:
    if os.path.isdir(f'{MAIN}/{src}') and not os.path.isdir(f'{REPO}/{dst}'):
        print(f'copying {src} -> {dst} (can take a while)...')
        os.makedirs(f'{REPO}/{dst}', exist_ok=True)
        rc, out = sh(f"rsync -a --max-size=600M {excl} '{MAIN}/{src}/' '{REPO}/{dst}/'")
        if rc: print(out[-2000:])
print(sh(f'du -sh outputs/*', cwd=REPO)[1])

# 4. Run every table generator with its documented arguments.
A = 'thesis_report/figures/auto'
RUNS = [
    'python -m scripts.aggregate_stochastic_equilibrium --log outputs/full_run/experiment.log',  # must precede 5_2, which reads its CSV
    'python -m scripts.aggregate_section_5_2',
    'python -m scripts.aggregate_section_5_3',
    'python -m scripts.aggregate_section_5_4',
    'python -m scripts.aggregate_section_5_5',
    'python -m scripts.aggregate_section_5_6',
    'python -m scripts.aggregate_section_5_7',
    'python -m scripts.aggregate_pi_store_trajectory --run_dir outputs/full_run --cycle_0_dir outputs/cycle_0',
    'python -m scripts.aggregate_baseline_pooled --baselines_dir outputs/baselines_phase1e --caem_eval_dir outputs/full_run/eval --chm_json outputs/baselines_phase1e/chm_comparison.json',
    'python -m scripts.baseline_sig_tests_v2 --caem_cycle 5 --out_tex ' + A + '/tab_sig_test.tex',
    'python -m scripts.baseline_sig_tests_v2 --caem_cycle 2 --out_tex ' + A + '/tab_sig_test_c2.tex',
    'python -m scripts.aggregate_tier_chm --run_dir outputs/full_run',
    'python -m scripts.aggregate_calibration_trajectory --run_dir outputs/full_run --cycle_0_dir outputs/cycle_0 --csv_out /tmp/calib_traj.csv --tex_out ' + A + '/tab_calibration_trajectory.tex',
    'python -m scripts.aggregate_three_headline_decomposition',
    'python -m scripts.aggregate_memory_store_trajectory',
    'python -m scripts.aggregate_per_tier_em',
    'python -m scripts.aggregate_ces_per_system',
    'python -m scripts.aggregate_survival_distribution',
    'python -m scripts.aggregate_decision_diagnostics',
    'python -m scripts.generate_session5_artifacts --out_figures ' + A + ' --out_tables ' + A,
    'python -m scripts.make_tables outputs/full_run -v',
    'python -m scripts.retrieval_utility_classifier --help',  # only to log its options
]
os.makedirs(RESULT, exist_ok=True)
log = [f'# regen run {datetime.datetime.now().isoformat()}  artefacts={MAIN}\n']
for cmd in RUNS:
    try:
        rc, out = sh(cmd, cwd=REPO, timeout=3600)
    except subprocess.TimeoutExpired:
        rc, out = 'TIMEOUT', ''
    print(f'[{rc}] {cmd}')
    log.append(f'\n===== [{rc}] {cmd}\n{out[-6000:]}')

# make_tables writes its .tex next to its CSV inputs; copy those over the thesis copies.
for t in glob.glob(f'{REPO}/outputs/full_run/*.tex'):
    if os.path.exists(f'{REPO}/{A}/' + os.path.basename(t)):
        shutil.copy(t, f'{REPO}/{A}/')

# 5. Diff regenerated tables against the committed ones and save everything to Drive.
_, diff = sh(f'git diff --word-diff=plain -U0 -- {A}', cwd=REPO)
_, stat = sh(f'git diff --stat -- {A}; git status --short -- {A}', cwd=REPO)
_, files = sh('find outputs -maxdepth 3 | sort', cwd=REPO)
open(f'{RESULT}/regen_diff.txt', 'w').write(diff)
open(f'{RESULT}/regen_stat.txt', 'w').write(stat)
open(f'{RESULT}/regen_log.txt', 'w').write(''.join(log))
open(f'{RESULT}/outputs_tree.txt', 'w').write(files)
shutil.copytree(f'{REPO}/{A}', f'{RESULT}/auto_regenerated', dirs_exist_ok=True)
print('\nDone. Results in MyDrive/caem_regen_check/ :')
print(stat)
