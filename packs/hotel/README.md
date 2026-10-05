# Hotel pack

A synthetic hotel revenue-management model and everything an agent needs to work on it. Apache-2.0, like optlens; all
data is generated and seeded.

| file | what it is |
|---|---|
| `generate.py` | the model builder: the booking model (`--out`), its linked variant (`--linked`), and the two-stage pipeline (`--pipeline DIR`: the rate plan, solved, then the booking model at its rates) |
| `model.md` | the model document: every rule and number in business terms |
| `skill.md` | domain notes for an agent (the levers planners own, the helpers, how the two stages connect) |
| `helpers.py` | business views and checks for an optlens session: `data_check` (every input that differs from the model rebuilt from its documented parameters), `carry` (two-stage: a rate-plan version's prices and demand applied as changes onto a booking-plan version, so kept changes survive), `occupancy`, `revenue`, `rate_plan`, `describe` |
| `hooks.py`, `pack.json` | the pack for optchat, the planner chat agent built on optlens (see Contact in the main README). It loads the helpers, opens the rate plan in a two-stage session and declares that the rate plan feeds the booking plan; optchat then asks the planner before carrying a rate-plan change, and warns when an approved rate plan was never carried |
| `examples/two_stage_14n/` | one hotel, 14 nights: `rates.lp.gz`, `bookings.lp.gz`, `levels.json`, `params.json` |

```
python packs/hotel/generate.py --nights 14 --out hotel_14n.lp            # one-stage booking model
python packs/hotel/generate.py --nights 14 --pipeline hotel_two_stage/   # rate plan, then bookings
```

With the optlens plugin in Claude Code, the helpers work in `run_python`:
`import sys; sys.path.insert(0, "packs/hotel"); import helpers as hotel; print(hotel.data_check(session, "v0"))`
(set `hotel.MODEL_FILE = MODEL_FILE` first so it finds `params.json` next to the model).

Tests: `cd packs/hotel && python -m unittest discover -s tests -t .`
