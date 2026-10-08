# optlens Claude Code plugin

The optlens MCP server (`optlens-mcp`, 22 tools) plus the `optlens` skill, which carries the same method the server
sends every MCP client (see "Use it with your agent" in the [README](../README.md)).

```bash
pip install "optlens[scip,mcp]"   # add [gurobi] if you have a license
claude plugin marketplace add jjd-lab/optlens
claude plugin install optlens@optlens
```

In a session, `/plugin marketplace add jjd-lab/optlens` then `/plugin install optlens@optlens` does the same. To try a checkout without installing,
start Claude Code with `claude --plugin-dir plugin` from the repository root.

Where GitHub cannot be reached (a proxy, a VPN), install from a copy of the repository instead:
`pip install "<folder>[scip,mcp]"` and `claude plugin marketplace add <folder>`. With `[gurobi]`, install the gurobipy
major version your Gurobi accepts (`pip install "gurobipy==12.*"` for a version 12 Compute Server): a newer client is
rejected, and `open_model` then says Gurobi cannot run and uses HiGHS or SCIP.

**Permissions.** In Claude Code's default mode each optlens tool asks on first use. To allow them all, run
`/permissions` and add `mcp__plugin_optlens_optlens` to the allowed tools (every tool of the plugin's server, including
`run_python`, which runs the agent's code; allow the tools by name, such as `mcp__plugin_optlens_optlens__open_model`,
to keep `run_python` asking).

The server (`optlens-mcp`) must be on the PATH Claude Code sees. On macOS, keep the venv out of folders that iCloud
syncs (`~/Documents`, `~/Desktop`): iCloud flags files hidden, and Python 3.13 skips hidden `.pth` files, so an
editable install's `optlens-mcp` cannot import the package (`chflags nohidden` on the `.pth` files clears it).
`OPTLENS_SOLVER` picks the solver ([Solvers](../README.md#solvers)).

What it adds to Claude Code:
- the optlens tools, including `add_model` and `compare_models` for several models or scenarios side by side,
  `save_model_context` for one saved description of each model (`.optlens/context/`), and `run_python`, a persistent
  Python process with the engine preloaded as `session` (the model loaded once; several steps per call; its versions
  shared with the other tools'). Solves stop at 45 s unless the agent asks for more (up to 95 s, when a solve was still improving): the
  plugin sets `OPTLENS_CALL_LIMIT` to 110, under the 2 minutes past which Claude Code moves a call to the
  background, and its server's `timeout` to 10 minutes, which overrides a shorter
  `MCP_TOOL_TIMEOUT` (some environments set 60 s). Set `OPTLENS_CALL_LIMIT` (seconds, 30 to 590) before starting
  `claude` to change it; `open_model` states the limits in force, and results say when a limit cut them short;
- the `optlens` skill: how to diagnose, explain and compare, the same text as the server's instructions.

**Security.** `run_python` executes the code the agent writes with your permissions (its process starts without your
API keys and tokens; Claude Code asks before each call), and opening a `.py` model runs that file up to its first
solve call. Open only models and code you trust; see [SECURITY.md](../SECURITY.md).

## Windows

The engine and the plugin work on Windows (Python 3.12 and 3.13 tested). `scripts/sandbox.sh` needs bash; in
PowerShell the same clean setup is these steps (`$R` is the optlens checkout, or its folder from a zip; `$T` the new
folder):

```powershell
$R = "C:\path\to\optlens"; $T = "$HOME\optlens-try"; New-Item -ItemType Directory "$T\project" | Out-Null
py -3.12 -m venv "$T\venv"
& "$T\venv\Scripts\python" -m pip install "$R[scip,mcp,pyomo]"   # or "optlens[scip,mcp,pyomo]"
$env:CLAUDE_CONFIG_DIR = "$T\claude-config"                     # a Claude config of its own: only this plugin
claude plugin marketplace add $R                                 # or jjd-lab/optlens
claude plugin install optlens@optlens
$ex = "$R\packs\hotel\examples\try_week"
Copy-Item "$ex\hotel_week.lp.gz", "$ex\params.json", "$R\packs\hotel\model.md" "$T\project"
```

To start it (and again for each later session, in a new PowerShell): set the same `CLAUDE_CONFIG_DIR`, put the venv
first on the PATH so Claude Code finds `optlens-mcp`, and start `claude` in the project:

```powershell
$T = "$HOME\optlens-try"; $env:CLAUDE_CONFIG_DIR = "$T\claude-config"; $env:PATH = "$T\venv\Scripts;$env:PATH"
Set-Location "$T\project"; claude                                     # claude --continue resumes
```

The questions are in `packs/hotel/examples/try_week/questions.md`. A Python script that uses the engine needs the
`if __name__ == "__main__":` guard (see the Quickstart in the README): Windows starts solver processes with `spawn`.
