---
model: claude-sonnet-5-5
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill]
tags: [try-week]
---

hotel_week.lp.gz is this week's 14-night booking plan for our hotel; model.md describes it. Night 5's group occupancy floor was typed as 1,200 instead of 120; assume it is corrected to 120. With that fixed, which business rule costs us the most revenue this week?
