# optlens Claude Code plugin

The optlens MCP server (`optlens-mcp`, 22 tools) plus the `optlens` skill, which carries the same method the server
sends every MCP client (see "Use it with your agent" in the [README](../README.md)).

```bash
pip install "optlens[scip,mcp] @ git+https://github.com/jjd-lab/optlens"   # add [gurobi] if you have a license
claude plugin marketplace add jjd-lab/optlens
claude plugin install optlens@optlens
```

In a session, `/plugin marketplace add jjd-lab/optlens` then `/plugin install optlens@optlens` does the same. To try a checkout without installing,
start Claude Code with `claude --plugin-dir plugin` from the repository root.

The server (`optlens-mcp`) must be on the PATH Claude Code sees. On macOS, keep the venv out of folders that iCloud
syncs (`~/Documents`, `~/Desktop`): iCloud flags files hidden, and Python 3.13 skips hidden `.pth` files, so an
editable install's `optlens-mcp` cannot import the package (`chflags nohidden` on the `.pth` files clears it).
`OPTLENS_SOLVER` picks the solver ([Solvers](../README.md#solvers)).

What it adds to Claude Code:
- the optlens tools, including `add_model` and `compare_models` for several models or scenarios side by side,
  `save_model_context` for one saved description of each model (`.optlens/context/`), and `run_python`, a persistent
  Python process with the engine preloaded as `session` (the model loaded once; several steps per call; its versions
  separate from the other tools'). Solves stop at 45 s unless the agent asks for more (up to 95 s, when a solve was still improving): the
  plugin sets `OPTLENS_CALL_LIMIT` to 110, under the 2 minutes past which Claude Code moves a call to the
  background, and its server's `timeout` to 10 minutes, which overrides a shorter
  `MCP_TOOL_TIMEOUT` (some environments set 60 s). Set `OPTLENS_CALL_LIMIT` (seconds, 30 to 590) before starting
  `claude` to change it; `open_model` states the limits in force, and results say when a limit cut them short;
- the `optlens` skill: how to diagnose, explain and compare, the same text as the server's instructions.

**Security.** `run_python` executes the code the agent writes with your permissions (its process starts without your
API keys and tokens; Claude Code asks before each call), and opening a `.py` model runs that file up to its first
solve call. Open only models and code you trust; see [SECURITY.md](../SECURITY.md).
