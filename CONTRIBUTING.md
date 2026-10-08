# Contributing

The most useful contribution is a model optlens gets wrong. Open an issue with the model file, or a smaller one that
shows the same thing. Include the question you asked, what optlens said, and what you expected.

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

Tests use `unittest`, not pytest. They run on HiGHS and SCIP. Tests that need Gurobi skip when gurobipy is not
installed. pip's gurobipy comes with a size-limited license, which is enough for them.

## Guidelines

- Every solve goes through `optlens.backends`, which enforces a hard time limit in a worker process. Never call a
  solver directly.
- A session on a solver the user chose does every step on that solver. HiGHS and SCIP may stand in for each other
  where only one can do a step, and the result says so.
- `optlens` imports only the standard library, its declared dependencies and its extras
  (`tests/test_import_rule.py`). It never imports an LLM SDK.
- Tool results are text for an agent or a person. Lead with the answer, and keep every number traceable to a solve.
- Keep the engine platform-agnostic (Linux, macOS, Windows).

By contributing you agree that your contribution is licensed under the Apache License 2.0, like the project.
