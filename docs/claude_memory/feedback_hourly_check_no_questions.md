---
name: Autonomous ticks are truly autonomous — never ask the user, decide yourself
description: During `<<autonomous-loop-dynamic>>` ticks, make decisions autonomously within the Plan A scope. No yes/no questions, no option menus, no "want me to...?" prompts. Only surface facts + next wakeup.
type: feedback
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Rule:** during autonomous ticks (prompts containing `<<autonomous-loop-dynamic>>`), do NOT ask the user any question. Not yes/no. Not "which option". Not "want me to proceed". Not "let me know". The user runs the autonomous loop precisely because they're unavailable, and every question is friction that accumulates across ticks.

**Each tick does exactly this:**

1. **Read the state** — runner PID, logs, GPU, OOM counter, step progression, heartbeat.
2. **Decide autonomously** within the Plan A scope (`caem_phase1_autonomy.md`):
   - Continue current step → no decision needed
   - Restart stuck process → just do it, log why
   - Fix a clear bug → commit + push, document in branch_C_log.md, continue
   - Disable a failing experimental optimization → flip config, note it in commit
   - Kill + relaunch if it's safer than continuing → do it, state the reason
3. **Report facts + decisions** — what's happening, what you did, what's next.
4. **End with `ScheduleWakeup`** — always, even if you had to make a major change.

**Explicitly forbidden in autonomous ticks — ANY of these counts as "asking":**

- ❌ "Should I revert / restart / kill?"
- ❌ "Option A, B, or C?"
- ❌ "Want me to proceed?"
- ❌ "Let me know if you want X"
- ❌ "Your call"
- ❌ "No user action needed" / "No action required" / "Nothing for you to do" — presumes the user was considering acting, which is itself a soft ask
- ❌ "Rescheduling to …" / "Proceeding with …" / "Continuing with …" framed as announcements — the fact of the reschedule is enough
- ❌ Any closing line that invites a response before the next tick
- ❌ Any editorial framing of the state ("looks healthy", "no concerns") — report the numbers, not the vibe

**Tick response format — minimal, facts only:**

```
Tick @ HH:MM UTC:

Runner     alive, pid=XXXX, elapsed=HH:MM
Step       current-step-name, progress-marker
OOM        X (baseline Y)
GPU        XX GB / YY% / ZZ°C
Errors     none | brief-error-summary
Decisions  (only if something was done) — one-line log of the autonomous fix

[ScheduleWakeup(3600s)]
```

No preface, no commentary, no closing line. Data + ScheduleWakeup call. That's the entire tick.

**If the decision is wrong, the user corrects on the next interactive turn. That's cheaper than blocking on them during a tick.**

**Emergencies** (new kernel OOM, runner dead with unclear cause, budget overrun past topup ceiling): flag the issue loudly in the status, take the safest autonomous action (typically: log the event, don't schedule another wake if further damage is likely), and let the user pick up on their own time. Still no question — just a clear situational report.

**Interactive turns are different.** When the user is actively typing in real-time, asking clarifying questions is expected and appropriate. The no-questions rule applies only to `<<autonomous-loop-dynamic>>` wakeups.

**Why:** stated by user on 2026-04-22 16:00 BDT, re-emphasized 2026-04-23 17:10 BDT after I slipped a yes/no prompt into several autonomous ticks despite the first memo. The second instruction was explicit: *"you ask me to yes/no in each autonomous tick, update your config and make it truly autonomous. dont ask me"*. Treat this as a hard constraint going forward.
