# optlens

A solver-agnostic debugger and explainer for LP and MILP models. Give it a model that is infeasible, or that solves
to an answer you don't trust, and it finds the conflicting constraints, the smallest changes that fix them, and what
each change costs. Every result comes back as text an agent or a person can read: in your own Python, in Claude Code
through the plugin, or in any MCP client.

optlens works on a model that is already built. It does not write models.

## Install

Python 3.11 or later. Not on PyPI yet; install from GitHub:

```bash
pip install "optlens[scip,mcp] @ git+https://github.com/jjd-lab/optlens"   # core: numpy, scipy, HiGHS
```

To run the quickstart below or the tests, clone the repository and install it from the clone
(`git clone https://github.com/jjd-lab/optlens && cd optlens && pip install ".[scip,mcp]"`).

| extra | adds |
|---|---|
| `scip` | SCIP (pyscipopt): a second open-source solver, with a native MIP IIS |
| `gurobi` | gurobipy: Gurobi, bring your own license |
| `pyomo`, `pulp` | load Pyomo and PuLP models |
| `mcp` | the MCP server `optlens-mcp` and the Claude Code plugin |

## Quickstart

```python
import optlens as od
from optlens.session import Session, Version

if __name__ == "__main__":  # solves run in worker processes, which start by re-importing this script
    md = od.load("tests/fixtures/ex_milp_tutorial__rhs_tighten__0.mps")  # LP or MPS
    s = Session({"v0": Version(md, None, "original model")})
    print(s.get_model_overview())   # size, status (here INFEASIBLE), constraint families
    print(s.compute_iis())          # the conflicting constraints, grouped by family
    print(s.fix_menu())             # the smallest change per family that restores feasibility
    print(s.modify_and_resolve(changes=[{"action": "set_rhs", "name": "resource[Monika]", "upper": 1}]))
```

Save it as a file and run it from the repository root; the `if __name__ == "__main__":` guard is needed on every
platform. Every `Session` method returns text, and `optlens.session.TOOLS` holds the matching JSON schemas for an
agent. A session also covers feasibility relaxation, what-if edits as versions, why-not questions, sensitivity and
marginal values, suspicious data values, and comparisons between versions or models.

A model built in Pyomo, gurobipy or PuLP loads directly, from the object or from the file that builds it. The file
is run without its `if __name__ == "__main__":` block and stops at its first solve call (`optimize()`, `solve()`),
whose model is the one read; nothing is solved, so a large gurobipy model loads on Gurobi's size-limited license.

```python
md = od.from_object(model)                      # a Pyomo ConcreteModel, gurobipy Model or PuLP LpProblem
md = od.load("my_model.py")                     # the model it solves, else its one model or build_model()
md = od.load("my_model.py:make_scenario")       # a named model, or a function with no arguments that returns one
```

LP files written by Gurobi (bracketed names, which HiGHS rejects) load without gurobipy.

Quadratic objectives (QP, MIQP) are supported: a convex QP solves on HiGHS, and a mixed-integer or non-convex one goes
to SCIP automatically. Every question works as for a linear model except sensitivity ranges (and shadow prices for a
non-convex objective). Quadratic constraints, indicator and other general constraints, and SOS
are rejected when the model loads. The full list of questions, formats and solvers: [optlens.dev/ask](https://optlens.dev/ask/).

## Solvers

HiGHS comes with the core; SCIP and Gurobi are extras. Pick one with `OPTLENS_SOLVER` (`highs`, `scip`, `gurobi`, or
`auto`, the default), `open_model`'s `solver` in the MCP server, or `Session(..., prefer=...)`. Under `auto` without
Gurobi, each model goes to HiGHS or SCIP (a larger MIP's first solve races both).

- **Every solve has a hard time limit.** Solvers do not always honour their own (SCIP once ran 862 s on a 30 s
  limit), so each solve runs in a worker process that is stopped at the limit.
- **Gurobi does every step itself.** A Gurobi session solves, computes IIS, relaxes, ranges and checks on Gurobi; no
  step is handed to another solver. If you chose Gurobi and it cannot run a model (not installed, or over the
  size-limited license), the tool stops and says so instead of switching. Under `auto` it uses Gurobi when gurobipy
  is installed and falls back to HiGHS or SCIP for a model Gurobi cannot run, and says so.
- **HiGHS and SCIP stand in for each other** where only one can do a step (HiGHS has no MIP IIS; SCIP does), or
  when one gives no verdict within its limit (the other tries once), and the result names the solver that did it.
- **Every IIS is checked.** A solver's IIS whose constraints are feasible on their own is rejected and rebuilt by
  removing constraints one at a time on the same solver. This guards against a known gurobipy 13.0.3 bug: `computeIIS` leaves out a
  one-variable row on a binary whose fractional limit rounds it to 0 (`160 open <= 100` with `open >= 1` returns
  `open >= 1` alone, which is feasible).
- **Gurobi is tested on the size-limited license only** (2,000 constraints and 2,000 variables). Large models under a full
  license, and Gurobi's IIS and relaxations at scale, are untested; reports are welcome.

On large models (hundreds of thousands of constraints) the IIS search starts near the conflict, from HiGHS's infeasibility
proof, instead of searching the whole model.

## Use it with your agent

Install as above (it puts `optlens-mcp` on your PATH), then connect once; the agent gets the 22 tools and the method
(open the model first, lead with the cause, every number from a solve, re-solve before recommending a fix) with them.

| agent | connect |
|---|---|
| **Claude Code** | `claude plugin marketplace add jjd-lab/optlens` then `claude plugin install optlens@optlens` (or the same as `/plugin` commands in a session) |
| **Codex CLI** | `codex mcp add optlens -- optlens-mcp` |
| **Cursor** | in `.cursor/mcp.json` (or `~/.cursor/mcp.json`): `{"mcpServers": {"optlens": {"command": "optlens-mcp"}}}` |
| **VS Code (Copilot)** | in `.vscode/mcp.json`: `{"servers": {"optlens": {"type": "stdio", "command": "optlens-mcp"}}}` |
| **Claude Desktop** | in `claude_desktop_config.json` (Settings > Developer > Edit Config): `{"mcpServers": {"optlens": {"command": "optlens-mcp"}}}` |
| **Any other MCP client** | run `optlens-mcp` as a stdio server |

The command must be on the PATH the agent sees; give the full path to `optlens-mcp` (for example `.venv/bin/optlens-mcp`)
if it is not. `OPTLENS_SOLVER` (`highs`, `scip`, `gurobi` or `auto`) chooses the solver once.

**Time limits.** Solves stop at 45 s, so a call answers within a minute. `OPTLENS_CALL_LIMIT` (seconds, at least 30)
is the most one tool call may take: a tool's `time_limit` may ask for up to 15 s less, and the agent asks for more only
when a solve stopped at its limit while still improving (results say whether it was). Set it to what your client waits
for one tool call; `open_model` states the limits in force, and a result cut short by one says so.

| client | waits for a tool call | `OPTLENS_CALL_LIMIT` |
|---|---|---|
| Claude Code (plugin) | 10 minutes, set by the plugin (calls past 2 minutes move to the background) | 110, set by the plugin |
| Claude Code (`claude mcp add`) | `MCP_TOOL_TIMEOUT` (about 28 hours unless set; some environments set 60 s) | 60 unless set; to allow more, add `"timeout": 600000` to the server's entry and set 110 |
| Claude Desktop, MCP TypeScript SDK clients | 60 s | 60 (the default) |
| other clients | see the client's settings | 60 unless the client allows more |

Large MIPs may not finish within any of these; results then report the plan found, its bound and its gap.

Then ask: *"Why is `plan.mps` infeasible, and what fixes it?"* The agent opens the model, writes its context once
(what each constraint and variable family means, as JSON in `.optlens/context/` that you can review and commit), and
works through the tools. For several steps or many solves it uses `run_python`, a persistent Python process with the
engine preloaded as `session` and the model loaded once.

The Claude Code plugin is the same server plus a skill that carries the same method ([plugin/README.md](https://github.com/jjd-lab/optlens/blob/main/plugin/README.md)).

**Security.** optlens runs code on your machine, with your permissions, and has no sandbox: `run_python` executes
the code the agent writes (its process starts without your API keys and tokens, but can read and write whatever your
user can; in Claude Code you approve each call), and opening a `.py` model runs that file up to its first solve call.
Open only models and code you trust. To report a vulnerability, see [SECURITY.md](https://github.com/jjd-lab/optlens/blob/main/SECURITY.md).

## Try it

To try the plugin without touching your own Claude Code setup, build a clean one in a folder of its own: its own venv
(optlens from GitHub), its own Claude config with only the optlens plugin, and a hotel week whose data load went wrong
([packs/hotel/examples/try_week](https://github.com/jjd-lab/optlens/blob/main/packs/hotel/examples/try_week/README.md)),
with five questions to ask:

```bash
git clone https://github.com/jjd-lab/optlens && optlens/scripts/sandbox.sh ~/optlens-try
~/optlens-try/start.sh          # log in on the first start; the questions are in ~/optlens-try/HOW-TO.md
```

`--project PATH` puts your own model there instead. It is a separate setup, not a security sandbox (see Security
above); delete the folder to remove it.

The same week in a clean Claude Code with the plugin (rendered from a recorded session; each tool call shows its real
duration):

![Claude Code with the optlens plugin finds why the week's hotel plan is infeasible, checks the fix, and ranks the business rules by the revenue they cost](https://raw.githubusercontent.com/jjd-lab/optlens/main/docs/plugin-agent-demo.svg)

## Hotel pack

[packs/hotel/](https://github.com/jjd-lab/optlens/blob/main/packs/hotel/README.md) is a synthetic hotel revenue-management model (room type × night × length of stay
× booking window × rate tier) with its generator, its document, domain notes and helpers. It builds models from a
few thousand to 837,000 constraints, for trying optlens on something realistic and large.

## Tests

```bash
python -m unittest discover -s tests -t .                             # the engine, from the repo root
cd packs/hotel && python -m unittest discover -s tests -t .           # the hotel pack
```

## Contact

Questions, feedback on your own models, or access to **optchat**, the chat agent built on optlens for business users
(available on request): **hello@optlens.dev**. Bugs and feature requests: GitHub issues.

A planner's session with optchat on the hotel pack's model (a 14-night plan, after a data load typed one night's group
target as 1,200 instead of 120). Each answer ends with the engine calls, time and cost it took; waiting time is cut
to two seconds:

![A planner asks why the week's plan is infeasible, approves the fix, and asks which rule costs the most revenue](https://raw.githubusercontent.com/jjd-lab/optlens/main/docs/planner-agent-demo.svg)

## Contributing

Issues with your own models are the most useful contribution; see
[CONTRIBUTING.md](https://github.com/jjd-lab/optlens/blob/main/CONTRIBUTING.md) and the
[code of conduct](https://github.com/jjd-lab/optlens/blob/main/CODE_OF_CONDUCT.md).

## License

Apache-2.0 ([LICENSE](https://github.com/jjd-lab/optlens/blob/main/LICENSE)); third-party credits in [NOTICE](https://github.com/jjd-lab/optlens/blob/main/NOTICE).
