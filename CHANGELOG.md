# Changelog

## 0.0.1 (unreleased, 2026-09-30)
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
