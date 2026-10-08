# optlens

A solver-agnostic debugger and explainer for LP and MILP models. Give it a model that is infeasible, or one that
solves to an answer you don't trust. optlens finds the conflicting constraints, the smallest changes that fix them, and
what each change costs. Every result is text that an agent or a person can read. Use it from your own Python, from
Claude Code through the plugin, or from any MCP client.

optlens works on a model that is already built. It does not write models.

## Install

Python 3.12 or later:

```bash
pip install "optlens[scip,mcp]"   # core: numpy, scipy, HiGHS
```

To run the quickstart below or the tests, clone the repository and install it from the clone
(`git clone https://github.com/jjd-lab/optlens && cd optlens && pip install ".[scip,mcp]"`).

Some networks cannot reach GitHub, for example behind a proxy or a VPN. There, copy the source over another way, such
as a zip of the repository. Install it with `pip install "<folder>[scip,mcp]"`, and add the Claude Code plugin with
`claude plugin marketplace add <folder>`.

| extra | adds |
|---|---|
| `scip` | SCIP (pyscipopt): a second open-source solver, with a native MIP IIS |
| `gurobi` | gurobipy: Gurobi, bring your own license. Its version must match your Gurobi. A Compute Server or token server rejects a newer client ("No compatible runtime available"), so install that major version, for example `pip install "gurobipy==12.*"`. `open_model` shows the installed gurobipy version. |
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

Save it as a file and run it from the repository root. Keep the `if __name__ == "__main__":` guard in any script that
solves. Solves run in worker processes. Windows starts them with `spawn`, and each one re-imports the script. Without
the guard, the script runs again in every solver process.

Every `Session` method returns text, and `optlens.session.TOOLS` holds the matching JSON schemas for an agent. A
session also covers feasibility relaxation, what-if edits as versions, why-not questions, sensitivity and marginal
values, suspicious data values, and comparisons between versions or models.

A model built in Pyomo, gurobipy or PuLP loads directly, from the object or from the file that builds it. optlens
runs the file without its `if __name__ == "__main__":` block and stops at the first solve call (`optimize()`,
`solve()`). It reads the model of that call. Nothing is solved, so a large gurobipy model loads on Gurobi's
size-limited license.

```python
md = od.from_object(model)                      # a Pyomo ConcreteModel, gurobipy Model or PuLP LpProblem
md = od.load("my_model.py")                     # the model it solves, else its one model or build_model()
md = od.load("my_model.py:make_scenario")       # a named model, or a function with no arguments that returns one
```

LP files written by Gurobi (bracketed names, which HiGHS rejects) load without gurobipy.

optlens supports quadratic objectives (QP, MIQP). A convex QP solves on HiGHS. optlens sends a mixed-integer or
non-convex one to SCIP automatically. Every question works as for a linear model, except sensitivity ranges. For a
non-convex objective, shadow prices don't work either. optlens rejects quadratic constraints, indicator and other
general constraints, and SOS when the model loads. [optlens.dev/ask](https://optlens.dev/ask/) lists every question,
format and solver.

## Solvers

HiGHS comes with the core, and SCIP and Gurobi are extras. Pick one with `OPTLENS_SOLVER` (`highs`, `scip`, `gurobi`,
or `auto`, the default), with `open_model`'s `solver` in the MCP server, or with `Session(..., prefer=...)`. Under
`auto` without Gurobi, each model goes to HiGHS or SCIP. A larger MIP's first solve races both.

- **Every solve has a hard time limit.** Solvers do not always honour their own limits. SCIP once ran 862 s on a
  30 s limit. So each solve runs in a worker process, and optlens stops the process at the limit.
- **Gurobi does every step itself.** A Gurobi session solves, computes IIS, relaxes, ranges and checks on Gurobi. It
  hands no step to another solver. Say you chose Gurobi and it cannot run a model, because it is not installed or the
  model is over the size-limited license. Then the tool stops and says so instead of switching.
  Under `auto`, optlens uses Gurobi when Gurobi can run here. It checks this once with a one-variable solve. It leaves
  out an installed gurobipy with no usable license, or a version your license server rejects, and `open_model` says
  why. For a model Gurobi cannot run, it falls back to HiGHS or SCIP and says so.
- **HiGHS and SCIP stand in for each other.** This happens where only one can do a step: HiGHS has no MIP IIS, and
  SCIP does. It also happens when one gives no verdict within its limit, and then the other tries once. The result
  names the solver that did the step.
- **Every IIS is checked.** Sometimes a solver's IIS has constraints that are feasible on their own. optlens rejects
  that IIS and rebuilds it on the same solver, by removing constraints one at a time. This guards against a known
  gurobipy 13.0.3 bug. `computeIIS` leaves out a one-variable row on a binary whose fractional limit rounds it to 0.
  For example, `160 open <= 100` with `open >= 1` returns `open >= 1` alone, which is feasible.
- **Gurobi is tested on small models and on one large one.** The large one is a 279k-row hotel model, tested for
  solves, sensitivity, IIS, relaxations and repair menus. Other large models are untested, and reports are welcome.

On large models, with hundreds of thousands of constraints, the IIS search starts near the conflict. It starts from
HiGHS's proof of infeasibility instead of searching the whole model.

## Use it with your agent

Install as above, which puts `optlens-mcp` on your PATH. Then connect once. The agent gets the 22 tools and the
method that goes with them: open the model first, lead with the cause, take every number from a solve, and re-solve
before recommending a fix.

| agent | connect |
|---|---|
| **Claude Code** | `claude plugin marketplace add jjd-lab/optlens` then `claude plugin install optlens@optlens` (or the same as `/plugin` commands in a session) |
| **Codex CLI** | `codex mcp add optlens -- optlens-mcp` |
| **Cursor** | in `.cursor/mcp.json` (or `~/.cursor/mcp.json`): `{"mcpServers": {"optlens": {"command": "optlens-mcp"}}}` |
| **VS Code (Copilot)** | in `.vscode/mcp.json`: `{"servers": {"optlens": {"type": "stdio", "command": "optlens-mcp"}}}` |
| **Claude Desktop** | in `claude_desktop_config.json` (Settings > Developer > Edit Config): `{"mcpServers": {"optlens": {"command": "optlens-mcp"}}}` |
| **Any other MCP client** | run `optlens-mcp` as a stdio server |

The command must be on the PATH the agent sees. If it is not, give the full path to `optlens-mcp`, for example
`.venv/bin/optlens-mcp`. `OPTLENS_SOLVER` (`highs`, `scip`, `gurobi` or `auto`) chooses the solver once.

**Time limits.** Solves stop at 45 s, so a call answers within a minute. `OPTLENS_CALL_LIMIT` (seconds, at least 30)
is the most time one tool call may take. A tool's `time_limit` may ask for up to 15 s less than that. The agent asks for
more time only when a solve stopped at its limit while still improving, and results say whether it was. Set the limit
to how long your client waits for one tool call. `open_model` states the limits in force, and a result that a limit
cut short says so.

| client | waits for a tool call | `OPTLENS_CALL_LIMIT` |
|---|---|---|
| Claude Code (plugin) | 10 minutes, set by the plugin (calls past 2 minutes move to the background) | 110, set by the plugin |
| Claude Code (`claude mcp add`) | `MCP_TOOL_TIMEOUT` (about 28 hours unless set; some environments set 60 s) | 60 unless set; to allow more, add `"timeout": 600000` to the server's entry and set 110 |
| Claude Desktop, MCP TypeScript SDK clients | 60 s | 60 (the default) |
| other clients | see the client's settings | 60 unless the client allows more |

Large MIPs may not finish within any of these limits. Results then report the plan found, its bound and its gap.

Then ask: *"Why is `plan.mps` infeasible, and what fixes it?"* The agent opens the model and writes its context
once. The context says what each constraint and variable family means. It is JSON in `.optlens/context/`, which you
can review and commit. Then the agent works through the tools. For several steps or many solves it uses `run_python`.
That is a persistent Python process with the engine preloaded as `session` and the model loaded once.

The Claude Code plugin is the same server plus a skill that carries the same method ([plugin/README.md](https://github.com/jjd-lab/optlens/blob/main/plugin/README.md)).

**Security.** optlens runs code on your machine, with your permissions, and has no sandbox. `run_python` executes
the code the agent writes. Its process starts without your API keys and tokens, but it can read and write whatever
your user can. In Claude Code you approve each call. Opening a `.py` model runs that file up to its first solve call.
Open only models and code you trust. To report a vulnerability, see [SECURITY.md](https://github.com/jjd-lab/optlens/blob/main/SECURITY.md).

## Try it

To try the plugin without touching your own Claude Code setup, build a clean one in a folder of its own. The folder
gets its own venv with optlens from GitHub, and its own Claude config with only the optlens plugin. It also gets a
hotel week whose data load went wrong
([packs/hotel/examples/try_week](https://github.com/jjd-lab/optlens/blob/main/packs/hotel/examples/try_week/README.md)),
with five questions to ask:

```bash
git clone https://github.com/jjd-lab/optlens && optlens/scripts/sandbox.sh ~/optlens-try
~/optlens-try/start.sh          # log in on the first start; the questions are in ~/optlens-try/HOW-TO.md
```

It needs bash, Python 3.12+ and Claude Code, on macOS or Linux. On Windows, use WSL or the PowerShell steps in
[plugin/README.md](https://github.com/jjd-lab/optlens/blob/main/plugin/README.md#windows). `--project PATH` puts your
own model there instead. It is a separate setup, not a security sandbox (see Security above). Delete the folder to
remove it.

Here is the same week in a clean Claude Code with the plugin. The animation comes from a recorded session, and each
tool call shows its real duration:

![Claude Code with the optlens plugin finds why the week's hotel plan is infeasible, checks the fix, and ranks the business rules by the revenue they cost](https://raw.githubusercontent.com/jjd-lab/optlens/main/docs/plugin-agent-demo.svg)

## Hotel pack

[packs/hotel/](https://github.com/jjd-lab/optlens/blob/main/packs/hotel/README.md) is a synthetic hotel
revenue-management model: room type × night × length of stay × booking window × rate tier. It comes with its
generator, its document, domain notes and helpers. It builds models from a few thousand to 837,000 constraints, so you
can try optlens on something realistic and large.

## Tests

```bash
python -m unittest discover -s tests -t .                             # the engine, from the repo root
cd packs/hotel && python -m unittest discover -s tests -t .           # the hotel pack
```

## Contact

Write to **hello@optlens.dev** with questions, feedback on your own models, or a request for access to **optchat**.
optchat is the chat agent built on optlens for business users. For bugs and feature requests, open a GitHub issue.

Here is a planner's session with optchat on the hotel pack's model, a 14-night plan. A data load typed one night's
group target as 1,200 instead of 120. Each answer ends with the engine calls, time and cost it took. We cut the waiting
time to two seconds:

![A planner asks why the week's plan is infeasible, approves the fix, and asks which rule costs the most revenue](https://raw.githubusercontent.com/jjd-lab/optlens/main/docs/planner-agent-demo.svg)

## Contributing

Issues with your own models are the most useful contribution. See
[CONTRIBUTING.md](https://github.com/jjd-lab/optlens/blob/main/CONTRIBUTING.md) and the
[code of conduct](https://github.com/jjd-lab/optlens/blob/main/CODE_OF_CONDUCT.md).

## License

Apache-2.0 ([LICENSE](https://github.com/jjd-lab/optlens/blob/main/LICENSE)). Third-party credits are in
[NOTICE](https://github.com/jjd-lab/optlens/blob/main/NOTICE).
