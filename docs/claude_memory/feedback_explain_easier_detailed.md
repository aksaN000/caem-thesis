---
name: Explain in easier words and in detail
description: When the user asks "what happened" or for an explanation, prefer plain-English step-by-step exposition over dense technical jargon — but stay detailed (don't oversimplify or skip mechanics).
type: feedback
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
When the user asks for an explanation of what happened during a session — or for any walk-through of complex events — explain in **easier words** AND **in detail**. Both at the same time.

**Why:** User explicitly asked for this on 2026-04-27 after the Cycle-1 calibration-drift / OOM-patches debugging session. They want to understand the full mechanism, not just a surface summary, but they don't want to wade through dense thesis-style prose to get there.

**How to apply:**
- Plain-English vocabulary first; technical terms only after they've been explained at least once
- Walk events in chronological order when narrating "what happened"
- Use concrete numbers (memory counts, timestamps, file sizes) — they help anchor the explanation
- Use short paragraphs and step-by-step structure (1, 2, 3 / "first", "next", "then")
- Tables and bullet lists are welcome; jargon-dense LaTeX-style prose is not
- Keep mechanism detail (why a thing happened the way it did), strip unnecessary caveats and academic hedging
- Length: as long as needed for clarity, NOT as short as possible
- Counter-example to avoid: a one-paragraph TL;DR followed by silence. The user wants the actual story.
