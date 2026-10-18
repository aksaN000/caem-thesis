---
name: feedback_thesis_skimmable_structure
description: "Thesis prose must be skimmable: when a passage lists parallel items (First/Second/Third, three families, etc.), mark them visibly, choosing the device that suits the text."
metadata:
  node_type: memory
  type: feedback
  originSessionId: 9bbed419-78c6-48fb-ad84-30c1f48abc6e
  modified: 2026-10-16T07:11:04.980Z
---

Aksan (2026-10-16, ch1/ch2 read-through): in ch1 sec:stateless-gap the three shortcomings were plain paragraphs starting "First, Second, Third"; he wants "reader should be able to skim easily ... always use suitable method for different parts according to need of the text."

**Why:** examiners skim; parallel items buried in running prose are easy to miss.

**How to apply (pick per passage, don't apply one device everywhere):**
- Parallel items that each take a paragraph → bold run-in lead-in naming the item, ENDING IN A COLON ("\textbf{No memory:} Standard inference ..."). Never a full stop: Aksan (2026-10-16) said the "No memory." form looks AI-generated. Fixed ch1+ch3; ch4 L306/L322 and ch5 L613–618 still to fix.
- Short parallel items (a sentence or two each) → itemize/enumerate.
- Subsubsection headings only when each item spans several paragraphs or is cross-referenced; otherwise they clutter the TOC.
- Check that the section title still names what the items are (e.g. three shortcomings shouldn't sit under a title naming only one).
- Related: [[feedback_thesis_writing_style]], [[feedback_chapter_review_workflow]].

**Figures and captions (Aksan, 2026-10-16, ch4):** add figures where text is hard to follow (TikZ, code-verified; CAEM book figures may be adapted but must be re-checked against the code: the book shows the ablated Qwen judge and unverified "deployed points"). Keep captions to one or two sentences; move the detail into the body text near the first reference. Render pages with pymupdf and look at them before committing.
