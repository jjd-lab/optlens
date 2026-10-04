# optlens Claude Code plugin

Install the engine with its MCP server, then load the plugin:

```bash
pip install -e ".[scip,mcp]"          # from the repository root; add [gurobi] if you have a Gurobi license
claude --plugin-dir plugin
```

The server (`optlens-mcp`) must be on the PATH Claude Code sees. On macOS, keep the venv out of folders that iCloud
syncs (`~/Documents`, `~/Desktop`): iCloud flags files hidden, and Python 3.13 skips hidden `.pth` files, so an
editable install's `optlens-mcp` cannot import the package (`chflags nohidden` on the `.pth` files clears it).
`OPTLENS_SOLVER` (highs, scip, gurobi, or auto,
the default) picks the preferred solver; auto uses Gurobi when gurobipy is installed and falls back to HiGHS or SCIP
per model when Gurobi cannot run it.

What it adds to Claude Code:
- the optlens tools (MCP server), including `add_model` and `compare_models` for several models or scenarios side by
  side, `save_model_context` for one saved description of each model (`.optlens/context/`), and `run_python`, a
  persistent Python process with the engine preloaded as `session` (the model loaded once; several steps per call,
  the cheapest setup in our tests; 50 s per call, its versions separate from the other tools');
- the `optlens` skill: how to diagnose, explain and compare;
- the `optlens:model-analyst` subagent (optional): analyzes one open model for the questions it is given and reports
  every number with its source. By default the main conversation answers itself; it sends one analyst per model, in
  parallel, only when several models each need their own multi-step analysis, with a brief that carries the
  objective, context, output and boundaries (after Anthropic's "How we built our multi-agent research system"). Analysts share the server; calls that open models are serialized, the rest run in
  parallel. Each tool call stops in time for Claude Code's 60 s call timeout (solves 45 s, large IIS searches 40 s;
  results say when a search stopped short).
