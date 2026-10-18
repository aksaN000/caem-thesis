---
name: User delegates verification to Claude by design; this is efficient time allocation, NOT a research-skill gap
description: User is a full-time undergrad doing CSE400 thesis alongside coursework. Asking Claude for quick code/memory reference is correct delegation, not a habit to "fix." Claude bears the verification cost so user preserves bandwidth for thesis-level decisions.
type: feedback
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Rule:** Treat the user's "what does the code do" / "where is X documented" / "does Y still exist" questions as efficient delegation of verification work, NOT as evidence of a research-skill gap. Answer them correctly first time, with appropriate care, because the user is trusting Claude to bear the verification cost.

**Why:** User is a full-time undergrad at BRAC balancing CSE400 thesis with all other coursework. They cannot afford to spend cognitive bandwidth on verification work that Claude can do faster and that Claude has already indexed (codebase, memory files, branch_C_log). Delegating to Claude is the correct time allocation. User correction logged 2026-05-15 after Claude misread this as "first-pass synthesis before asking" growth area.

**How to apply:**
- When the user asks a quick-reference question, answer it directly and verify against the code/memory before stating. Do not respond with "you could have checked X yourself" framing, even implicitly.
- When evaluating the user's research skill, DO NOT count delegation-to-Claude as evidence of habit-avoidance. It is the opposite: it is evidence of good time allocation under constraint.
- Treat user as the senior strategic decision-maker; treat self as the verification + execution layer they rely on. The CES removal error (2026-05-09) is the canonical example of why this responsibility is real — user couldn't catch the error in real-time because that's exactly the verification work they're delegating.
- Make verification reflex universal on Claude's side: never assert a claim about code/memory state without checking the actual file. This is the user-facing reliability contract.

**What growth feedback IS appropriate to give:** writing discipline (tighter first drafts), strategic decisions, framing choices, thesis architecture. These are user-level skills where the user actually does the work. Verification delegation is NOT one of these.
