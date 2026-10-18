---
name: Always prefer the cleaner architectural approach over hacky shortcuts
description: When given the choice between a cleaner design (proper API, well-typed, no special-casing) and a hackier shortcut (inline, ad-hoc, defensive), always pick the cleaner path. Aksan stated this preference 2026-05-19 during the RUC pipeline-wiring discussion, after rejecting the "skip p_ik for now, add later" shortcut and the "load a separate frozen base model" workaround in favour of the disable-adapter context manager.
type: feedback
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Rule.** Always prefer the cleaner architectural approach when multiple paths exist. Reject shortcuts that defer real work or hide structural issues behind ad-hoc workarounds.

**Why.** Aksan stated this preference 2026-05-19. Two examples from the same conversation:
- Proposed skipping p_ik at deployment ("add later"). Rejected — p_ik is the strongest feature, so dropping it changes the result, not just the timeline.
- Proposed loading a separate frozen Qwen for P(IK) hidden states (~6 GB extra GPU). Aksan implicitly preferred the disable_adapter context-manager approach (0 memory, proper peft API usage) when given the choice.

**How to apply.**
- When facing a design decision with a "shortcut" option and a "clean" option that costs ~30 min more, take the clean option without asking.
- When the clean option costs hours more than the shortcut, surface the trade-off explicitly with both costs and recommend the clean path unless there's a compelling reason otherwise.
- "Skip X for now" / "add later" / "TODO" are warning signs — they usually become permanent. Only acceptable when explicitly time-boxed by Aksan with a concrete deadline.
- Inline-import-inside-function and other "minimum surgical" patterns are fine for one-off fixes but not for things that will be touched repeatedly. Use top-level imports and proper module structure.
- Explicit references (self.X = X stored at __init__) > digging into internals (self.verifier._private_attr). Even if it adds a few lines.
- Hand-rolled workarounds > "let's just check if it works without the feature first." Build the right thing.

**Reverse direction.** If choosing the clean approach risks breaking running pipelines or costs >4 hours of dev time with high failure risk, surface the trade-off and ask Aksan to weigh in. Don't blindly take the clean path when it's actually risky to the timeline.
