# Try it: a hotel week with a data slip

One hotel, 14 nights, from the hotel pack's linked variant. `model.md` in the pack describes every rule. During a
data load, someone typed one input wrong, and the plan comes back infeasible. Five planner questions in
[questions.md](questions.md) walk through what the optlens plugin does with it. The agent finds the cause, values the
rules, tries a change, explains a decision and checks the data.

| file | what |
|---|---|
| `hotel_week.lp.gz` | the model as loaded, with the slip |
| `params.json` | the documented parameters, which the pack's `data_check` rebuilds the model from |
| `questions.md` | the five questions |

Try it in a clean Claude Code that leaves your own setup alone. From a clone of this repository, run
`scripts/sandbox.sh ~/optlens-try`. It builds a folder with its own venv, Claude config, the plugin and this example,
and prints how to start it. Or open the files with the plugin in your own Claude Code.

It was made with this command, and then one number was changed by hand (below):
`python packs/hotel/generate.py --nights 14 --hotels 1 --linked --out hotel_week.lp --params params.json`.

<details>
<summary>The slip (read it after your session)</summary>

Night 5's group occupancy floor, `group_occupancy(d5)`, is 1,200 rooms instead of 120. In `run_python`, the pack's
`data_check` confirms it against `params.json`:

```python
import sys; sys.path.insert(0, "<path to packs/hotel>"); import helpers as hotel
hotel.MODEL_FILE = MODEL_FILE; print(hotel.data_check(session, "v0"))
```

</details>
