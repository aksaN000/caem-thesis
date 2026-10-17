# Prompt for the agent on the rented GPU (final-version, 2026-10-17)

Paste the block below into Claude Code on the rental, from `/workspace/caem`.

```
You are running the final-version baseline re-run for the CAEM thesis on this rented GPU. Everything is prepared;
your job is to execute it carefully, not to change the design.

1. Read scripts/rental/final_version.sh fully, then the "Execution" and "M-A" sections at the top of
   defense/thesis_report/ref_check/MASTER_ISSUES.md.
2. Work inside tmux (session name: final). Run the stages one at a time, in this order, with
   `bash scripts/rental/final_version.sh <stage>`: preflight, env, data, search, baselines, flare, b6, gate,
   verify, pack. Do not use `all`; check each stage's log before starting the next.
3. Before each GPU stage, state the expected count (300 questions per benchmark, 5 benchmarks; B3/B4 TruthfulQA
   only) and confirm it in the log afterwards.
4. Stop and report (do not work around it) if: the search check share is below 0.95; the gate stage fails; any
   run logs "retried ... failed batch answers" more than a few times; the exact-search cache shows live searches
   for B1/B3/B4 or the verifier pass (FLARE's later searches are expected to be live); anything crashes twice.
5. Do not edit code, prompts, thresholds or scoring. If a genuine bug blocks a stage, describe it with the log lines
   and wait. Do not delete or overwrite anything under outputs/ except by re-running a stage.
6. Never print tokens or keys. Report clock times in Dhaka time (BDT).
7. When pack finishes, report: per-stage wall time, the search check numbers, the gate output, the cache hit/live
   counts, and the path and size of outputs/rerun_v2_results.tgz.
```
