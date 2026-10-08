---
type: llm
---

PASS if the answer explains the conflict (row 2's demand of 130 cannot be met because production is capped through rows 14 and 16 and the capacity rows such as 4 and 6) and gives either fix with its effect from a solve with the fix applied (a re-solve, or the plan a fix menu reports for it):
- the smallest single change, about 19.15: row 2's right-hand side down to about 110.85, row 14's up to about 24.15, or row 16's up to about 21.15; or
- row 4's limit of 8 read as a likely typo for 80,000, because its sibling rows hold 80,000, 70,000 and 70,000.
Listing more rows in the conflict is fine.
FAIL if it proposes deleting constraints, gives neither fix, or gives a fix with no solve behind it.
