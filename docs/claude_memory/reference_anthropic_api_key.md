---
name: anthropic-api-key-location
description: "User's Anthropic API key for scripts/rescore_truthfulqa.py and any other CAEM API call lives in ~/.bashrc as ANTHROPIC_API_KEY env var"
metadata: 
  node_type: memory
  type: reference
  originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---

The user's Anthropic API key is set as the `ANTHROPIC_API_KEY` environment variable, persisted via an `export` line in `~/.bashrc`. Any shell that sources bashrc (default interactive shells on this WSL2 host) has it available.

**Where:** `~/.bashrc` (line near the bottom). Login shells pick it up automatically.

**When to use:**
- `scripts/rescore_truthfulqa.py` — Haiku-judged TruthfulQA EM rescoring (memory: [[caem_truthfulqa_rescore_findings]])
- Any other CAEM API call that needs to hit `claude-haiku-4-5-20251001` or similar Anthropic models

**Usage pattern:**
```bash
source ~/.bashrc  # ensures key is loaded in non-interactive shells
python -m scripts.rescore_truthfulqa --files <path>
```

Or for one-off Python:
```python
import anthropic
client = anthropic.Anthropic()  # reads env var automatically
```

**Authorisation:** The user explicitly shared this key during the 2026-06-11 baseline rescore session with the instruction "keep it in mind. if needed to use later from now on", so it's available for any future Anthropic API call this CAEM project needs without re-asking. Do NOT echo the key value back in conversation; just confirm via `[ -n "$ANTHROPIC_API_KEY" ] && echo "key present"`.

**Rotation TODO:** the key is in plaintext in the 2026-06-11 conversation transcript. After the Vast Phase 2 rerun completes, remind the user to rotate it at https://console.anthropic.com/settings/keys and update `~/.bashrc` locally + the same value on any Vast instance that has it cached.
