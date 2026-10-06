# Changelog

## 0.0.1 (unreleased, 2026-09-30)
- Try it without touching your own setup: `scripts/sandbox.sh DIR` builds a clean Claude Code with the plugin in one
  folder (its own venv from GitHub, its own Claude config, a project, `start.sh`), by default with the hotel pack's
  new `examples/try_week` (a week whose data load typed one group target as 1,200 for 120) and five questions.
- The plugin's tools and `run_python` share versions and solves: a version a tool made is in `run_python`'s `session`
  with its solve, and one made there is in the tools. The server saves the open model's versions to
  `.optlens/sessions/` and restores them when the same model (same numbers) is opened again, so a resumed conversation
  keeps its scenarios; versions made on earlier data of the model are listed, not restored.
- A solve with no verdict (no plan, and not proven infeasible or unbounded) is tried once on the other open-source
  solver within the call's time, which then serves that version: HiGHS ran 120 s without deciding an infeasible
  4.7k-row blending LP that SCIP decides in 5 s. An IIS cut short at more than 200 rows points to
  `feasibility_relaxation`.
- `suspicious_values` flags an objective cost an order of magnitude off its group's other costs (a freight of 0.0084
  among 0.6-3.2), one flag per value however many columns carry it.
- `run_python`'s description names the result fields (`session.solved(v)`: `.status`, `.obj`, `.bound`, `.gap`, `.x`,
  duals), `md.lp_relaxation()` and that a model is frozen; the skill keeps a MIP effect to its proven range (an LP
  relaxation's change is only an indication). A model that imports a missing package says which Python to install it
  into.
- `run_python`'s worker starts from the server's solve of the original model instead of solving it again (in E62 a
  90 s what-if after the worker's own 45 s base solve overran the 100 s call), and its `time_limit` may ask for up to
  15 s less than the call. Thousands of `set_rhs` changes apply 28x faster: a check compared the row count with `is not`
  and rebuilt the name sets for every change (6,240 changes took about 42 s on a 24k-row model, now 2 s).
- The plugin's call limit is 110 s (a `time_limit` up to 95 s): with 300 the agent asked for the maximum on its first
  what-if although the base solve showed more time would not help, Claude Code moved the call to the background past
  2 minutes, and the answer did not come. `run_python`'s description says `session` methods start large MIPs from a
  plan while `od.BACKENDS[...].solve` solves from scratch (an agent solved cold that way and got $83.7M against $27.7M).
- The default solve limit stays 45 s whatever `OPTLENS_CALL_LIMIT` says; that is now the most a tool's `time_limit` may
  ask for. In a plugin trial a 300 s default made `open_model` run past 2 minutes, Claude Code moved it to the
  background, and the agent started its own solve of the same model meanwhile. `modify_and_resolve` takes a
  `time_limit` through the MCP server. A solve from a starting plan says whether the solver improved on it; the
  instructions read the base solve as the probe (improving: a longer `time_limit` may help; not: report ranges) and
  say to wait for a call moved to the background instead of solving the model another way.
- A MIP of 5,000 rows or more starts from a plan built from its LP relaxation (`optlens.starting_plan`): the
  integers the LP uses rounded up and the LP solved with them fixed, else the integers it leaves whole kept and the
  rest solved as a small MIP. It takes at most a fifth of the solve limit (the solve gets the rest); the result says
  when a start was used, and keeps the start when the solver returns nothing better. On a 24k-row setup-run MILP the
  solvers' own plans stayed at a 68 % gap for 300 s; from the start the gap is 3 % at 45 s (M08, M09). Backends take
  `start=` (HiGHS, SCIP, Gurobi).
- `OPTLENS_CALL_LIMIT` (seconds per MCP tool call, default 60, at least 30) sets the server's limits together: solves
  15 s less, a large MIP's IIS search 20 s less, `run_python` calls 10 s less, and `run_python`'s session solves inside
  that. The Claude Code CLI waits longer than 60 s for a tool call, so its users can give slow MIP solves minutes; the
  plugin passes the variable through. The 60 s default keeps clients that stop at 60 s (Claude Desktop) working.
  The plugin's default is 300, and its server's `timeout` is 10 minutes: Claude Code stops a tool call at
  `MCP_TOOL_TIMEOUT`, which an environment may set to 60 s (a cloud session did, and `open_model` timed out while its
  base solve ran); the server's own `timeout` overrides it. The instructions tell an agent whose call timed out to
  have the user lower `OPTLENS_CALL_LIMIT` rather than solve the model another way. A tool's `time_limit`, and the
  engine's own search budgets (an LP's IIS search had 300 s), stay within the solve limit, and a lowered
  `time_limit` is said in the result; `open_model` states the limits and the tools' schemas give the real default.
- The server's instructions and the plugin's skill say to load the tools before the first call: Claude Code defers
  them, and agents that searched for bare names (`open_model`) found nothing, worked without the descriptions and spent
  3-8 calls per session on `dir()` and `inspect.signature` (in a 2026-10-05 trial, `save_model_context` failed its
  first call in all four runs).
- Found in a trial on a 53k-row MILP (2,470 binaries) that no solver finishes in minutes (2026-10-05):
  - A MIP result carries the solver's bound (`SolveResult.bound`, `.gap`; HiGHS, SCIP and Gurobi). A plan not proven
    optimal is reported with its bound and gap, and an objective change between two such plans with the range the
    optimal change must lie in: a what-if difference smaller than the solvers' uncertainty no longer reads as a
    result. A time limit without a plan says it is not proof that none exists (`try_options` printed the right fix
    as a bare TIME_LIMIT next to worse fixes with objectives).
  - `run_python`'s session solves within the call's timeout (the call timeout less 10 s, at most the usual 60 s):
    it solved for 60 s inside a 50 s call, so every MIP solve that ran to its limit restarted the process.
  - `fix_menu` returns within its time limit: the drop-family screen gets a third of it and the relaxations, now
    all in parallel with the unrestricted one, the rest (it took 137 s with a 45 s limit).
  - The plugin manifest has no `version`, so Claude Code takes the commit as the version and `claude plugin update`
    picks up every change (a pinned 0.0.1 kept users on their first copy).
- Found in a fresh-install trial on a 48k-row LP (2026-10-05):
  - A large LP's IIS starts near the conflict, as a large MIP's does: HiGHS's whole-model IIS ran 305 s without
    finishing on an LP whose conflict is 2 rows; the localized search finds it in 0.4 s.
  - `fix_menu` 86 s -> 9 s: `ModelData.col_index` looked names up with `tuple.index` (76 s in one add_row of 103k
    terms); it now keeps a name -> position map per names tuple.
  - `compare_versions` no longer fails when a version dropped or added a constraint: it compares the rows both
    versions have and lists the others.
  - `suspicious_values` ranks each tier farthest from typical first (by order of magnitude). It listed flags in file
    order and cut at 40, so a demand of 35,200 against a typical 485 came 40th, after values 4x their group median.
  - The MCP server checks each call's arguments against the tool's schema and, when they do not fit, returns the
    problems and the schema without running anything. Claude Code can defer a server's tools, so an agent may call
    one whose schema it never saw (it guessed `label`, `constraint` and `rhs` for `modify_and_resolve`).
    `try_options` checks its changes against the same change schema as `modify_and_resolve`.
  - The install lines: `claude plugin install` has no `--marketplace` option (the site's line failed); the plugin
    installs with `claude plugin marketplace add jjd-lab/optlens`, then `claude plugin install optlens@optlens`. The
    Claude Code CLI waits longer than 60 s for a tool call; the 45 s solve limit is for MCP clients that stop at 60 s.
- Quadratic objectives: a non-convex one goes to SCIP automatically (HiGHS returned an unexplained status), and its
  sensitivity report says why there are no shadow prices; Gurobi no longer fails on one (it read shadow prices
  Gurobi does not have for a non-convex QP). `ModelData.convex_objective()` checks the sense-adjusted quadratic term.
- One way to use optlens from an agent: the MCP server now sends the method (open the model first, lead with the
  cause, every number from a solve, re-solve before recommending, `run_python` for many steps) as its instructions,
  so Codex, Cursor, Copilot, Claude Desktop and any MCP client get it with the tools. The Claude Code plugin is the
  same server plus a skill with the same text, and installs by name from this repository's marketplace
  (`claude plugin marketplace add jjd-lab/optlens`, `claude plugin install optlens@optlens`). The `model-analyst`
  subagent is gone: delegation cost 2.5x and added no correctness.
- Solver choice is respected: a session on Gurobi does every step on Gurobi (IIS, LP-relaxation IIS, repairs,
  checks); no step is handed to HiGHS or SCIP. If the user chose Gurobi (`OPTLENS_SOLVER=gurobi` or `open_model`'s
  `solver`) and it cannot run a model (size-limited license, not installed), the tool stops and asks instead of
  switching; under the default "auto" it falls back to HiGHS or SCIP and says so. HiGHS and SCIP, both open source, may
  stand in for each other where only one can do a step (HiGHS has no MIP IIS), and the result names the solver used.
  `sensitivity_report` takes shadow prices and ranges from Gurobi in a Gurobi session.
- Sensitivity ranges fixed for rows with slack: HiGHS's ranging for such a row is not a range of its limit, and the
  report printed it as one (a `<= 3` row with activity 0 showed "valid for RHS in [0, 3]"; the limit can move anywhere
  from 0 up). Such a limit now ranges from the row's activity to infinity (upper) or from minus infinity to it
  (lower), as Gurobi reports; checked against re-solves on 11 LPs. The report also says that at a degenerate solution
  the shadow price can differ on each side of a limit, and that `marginal_value` re-solves to check a change.
- Every native IIS is checked: one whose rows are feasible on their own is rejected and rebuilt by removing rows one at
  a time on the same solver. gurobipy 13.0.3's `computeIIS` leaves out a one-variable row on a binary whose fractional
  limit rounds it to 0 (`160 open <= 100` with `open >= 1` returns `open >= 1` alone, which is feasible).
- The package `optlens`, with `pyproject.toml` and the optional extras `scip`, `gurobi`, `pyomo`, `pulp` and `mcp`;
  `optlens.__version__`; type hints marked with `py.typed`.
- The tool layer `optlens.session`: `Session`, `Version` and the `TOOLS` schemas, with the time limit and the default
  solver as constructor arguments.
- `compare_versions`: what differs between two solved versions, per family and per variable, against the optimal plan
  closest to the first version's (`optlens.closest`, one extra solve), so on models with many equal-cost plans it
  reports the forced changes, not the solver's tie-breaking. `why_not` reports how many variables changed and groups
  them by family and by the forced variable's indices.
- `marginal_value` prints the new bound, the objective and the plain change each way; sensitivity cost ranges are
  printed in words.
- Hard solve time limits work on Windows (the worker processes fall back from forkserver to spawn).
- `optlens.lpfile`: an LP-file reader with no solver library, so Gurobi-written `.lp` files (bracketed names, which
  HiGHS rejects) load without gurobipy; it matched Gurobi's reader on every LP file it was tested on.
- Pyomo models load directly: `optlens.from_pyomo(model)`, or `load("model.py")` / `load("model.py:name")` for the
  file that builds one (also in the MCP `open_model`); Pyomo's names are kept, ranges stay one row, fixed variables
  stay fixed columns. Extra `optlens[pyomo]`.
- gurobipy and PuLP models load too: `from_gurobipy`, `from_pulp` (PuLP 2.x and 4.x), `from_object` for any of the
  three. Loading a `.py` file stops at its first solve call and reads that model, so scripts that solve at module
  level load without solving. Extra `optlens[pulp]`.
- Saved model context: the host model writes a model's business meanings once (`save_model_context` in the MCP
  server), validated against the model's families and kept as JSON in `.optlens/context/` (or `OPTLENS_CONTEXT_DIR`);
  `open_model` shows it in the same words in every later session. One renderer (`optlens.context.render`) for the
  plugin and any agent.
- Several models in one session: `Session.add_model(path, name)` and `compare_models` (status, objective and the same
  per-family metrics side by side), offered by the MCP server; `compare_versions` works across models that share their
  variables.
- The MCP server runs tool calls in parallel (only calls that open models are serialized; the session allocates
  version ids under a lock) and keeps each call within Claude Code's 60 s timeout. `compare_models` with one name gives that model's summary, including the value of each
  one-row constraint (a budget's spend).
- Suspicious values on anonymous models: senses are compared within rows of the same shape and within runs of
  consecutive rows, and a row flagged by several groups gets the tightest group's reason.
