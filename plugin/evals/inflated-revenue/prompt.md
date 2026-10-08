---
model: claude-sonnet-5-5
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill]
tags: [harder]
---

hotel_week.lp is this week's 14-night booking plan for our hotel; model.md describes it and params.json holds the documented parameters. After today's data load the plan shows $587,251 of room revenue, which looks too good. Is anything in the data wrong, and what would the plan earn without it?
