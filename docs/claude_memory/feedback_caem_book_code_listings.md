---
name: feedback_caem_book_code_listings
description: "CAEM Book per-stage chapters should include code listings (caempy style), not prose-only"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---

In "The CAEM Book" (the LaTeX teaching companion under `caem/caem book/`), the user
approved keeping actual code listings in the per-stage deep-dive chapters. Ch1–3
were prose-and-diagrams only; Ch4 (Stage 1 u_pre) introduced the first `lstlisting`
(the u_token loop) using the `caempy` style defined in `preamble.tex`. User verdict:
"keep the code listings."

**Why:** the book's explicit purpose is to explain how CAEM works from the raw code;
showing a faithful, condensed excerpt of the actual mechanism is the natural
fulfilment of "build it from raw code" and "deeper than the sessions."

**How to apply:** include one or a few tight, faithful code excerpts per per-stage
chapter (Ch4 onward) where they illuminate the mechanism. Keep them ASCII-only
(the caempy literate map covers τ α θ ρ λ φ ≥ ≤ × → if needed), breaklines=true so
they never overflow. Still lead with intuition + precise math; the listing
supplements, it does not replace prose. See [[feedback_thesis_writing_style]] (the
THESIS bans code names in prose — the BOOK does not; different documents).
