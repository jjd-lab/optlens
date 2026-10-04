# Contributing

Thanks for helping. The most useful contribution is a model optlens gets wrong: open an issue with the model file (or
a smaller one that shows the same thing), the question you asked, what optlens said, and what you expected.

## Development setup

```bash
git clone https://github.com/jjd-lab/optlens && cd optlens
python -m venv .venv && .venv/bin/pip install -e ".[scip,mcp,pyomo,pulp]" ruff
```

## Checks before a pull request

```bash
.venv/bin/ruff check src tests packs
.venv/bin/python -m unittest discover -s tests -t .                       # the engine (from the repo root)
cd packs/hotel && ../../.venv/bin/python -m unittest discover -s tests -t .  # the hotel pack
```

Tests use `unittest`, not pytest. They run on HiGHS and SCIP; tests that need Gurobi are skipped without gurobipy
(pip's gurobipy comes with a size-limited license that is enough for them).

## Guidelines

- Every solve goes through `optlens.backends`, which enforces a hard time limit in a worker process; never call a
  solver directly.
- A session on a solver the user chose does every step on that solver. HiGHS and SCIP may stand in for each other
  where only one can do a step, and the result says so.
- `optlens` imports only the standard library, its declared dependencies and its extras
  (`tests/test_import_rule.py`); it never imports an LLM SDK.
- Tool results are text for an agent or a person: lead with the answer, keep every number traceable to a solve.
- Keep the engine platform-agnostic (Linux, macOS, Windows).

By contributing you agree that your contribution is licensed under the Apache License 2.0, like the project.
