---
name: Do not add Claude Co-Authored-By trailer to git commits
description: User does not want the "Co-Authored-By: Claude ..." trailer in commit messages. Omit it from all future commits.
type: feedback
originSessionId: 30992e2c-c245-439e-947d-20a565da17c6
---
Do NOT include `Co-Authored-By: Claude <noreply@anthropic.com>` (or any Claude variant) in git commit messages.

**Why:** User asked to remove it on 2026-04-18 after the first commit of the session included the trailer. Reason not stated, but most likely: he wants the commit history to look like solo authorship for thesis attribution, contribution graph aesthetics, or to avoid signalling AI involvement in repo metadata.

**How to apply:**
- Default Claude Code commit-message conventions include the Co-Authored-By trailer; explicitly REMOVE that line every time.
- Just write the commit message normally with no trailer at the end.
- Applies to all repos, not just caem-thesis.
- If unsure whether the user has changed his mind on a fresh repo, ask once — but the default should be "no trailer."
