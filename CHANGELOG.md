# Changelog

## 0.0.1 (unreleased, 2026-09-30)
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
- `compare_versions` diffs against the optimal plan closest to the first version's (`optlens.closest`, one extra
  solve), so on models with many equal-cost plans it reports the forced changes, not the solver's tie-breaking.
- `compare_versions`: what differs between two solved versions, per family and per variable; `why_not` reports how
  many variables changed and groups them by family and by the forced variable's indices.
- `marginal_value` prints the new bound, the objective and the plain change each way; sensitivity cost ranges are
  printed in words.
- Hard solve time limits work on Windows (the worker processes fall back from forkserver to spawn).
- CI: ruff, and the unit tests on Linux, macOS and Windows.
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
- Plugin subagent `optlens:model-analyst` (general purpose: one open model, the brief's questions and metrics, a fixed
  report with every number's source; it loads the optlens skill). The MCP server runs tool calls in parallel (only
  calls that open models are serialized; the session allocates version ids under a lock) and keeps each call within
  Claude Code's 60 s timeout. `compare_models` with one name gives that model's summary, including the value of each
  one-row constraint (a budget's spend).
- Suspicious values on anonymous models: senses are compared within rows of the same shape and within runs of
  consecutive rows, and a row flagged by several groups gets the tightest group's reason.
