---
name: Use Dhaka time (BDT, UTC+6) in communication
description: User is in Bangladesh; always report timestamps in Dhaka time (UTC+6 / Asia/Dhaka), not UTC, when talking to the user. Logs can still use UTC — this is for conversational messages.
type: feedback
originSessionId: 30992e2c-c245-439e-947d-20a565da17c6
---
Always use Dhaka time (BDT, UTC+6 / Asia/Dhaka) when communicating times/ETAs to the user in chat messages.

**Why:** User is in Bangladesh. UTC timestamps force mental math every time. He asked explicitly: "use dhakatime when talk."

**How to apply:**
- In conversational messages: report clock time in BDT (e.g., "20:20 BDT" not "14:20 UTC").
- In tool commands / logs / scripts / file contents: UTC is still fine (matches server defaults and tmux log timestamps). Don't rewrite logs or scripts to use BDT.
- Quick conversion: BDT = UTC + 6 hours.
