# optlens Claude Code plugin

The plugin is the optlens MCP server (`optlens-mcp`, 22 tools) plus the `optlens` skill. The skill carries the same
method that the server sends every MCP client (see "Use it with your agent" in the [README](../README.md)).

```bash
pip install "optlens[scip,mcp]"   # add [gurobi] if you have a license
claude plugin marketplace add jjd-lab/optlens
claude plugin install optlens@optlens
```

In a session, `/plugin marketplace add jjd-lab/optlens` then `/plugin install optlens@optlens` does the same. To try
a checkout without installing, start Claude Code with `claude --plugin-dir plugin` from the repository root.

Where GitHub cannot be reached, for example behind a proxy or a VPN, install from a copy of the repository instead.
Run `pip install "<folder>[scip,mcp]"` and `claude plugin marketplace add <folder>`.

With `[gurobi]`, install the gurobipy major version your Gurobi accepts, for example `pip install "gurobipy==12.*"` for
a version 12 Compute Server. Gurobi rejects a newer client. `open_model` then says Gurobi cannot run, and optlens uses
HiGHS or SCIP.

**Permissions.** In Claude Code's default mode, each optlens tool asks on first use. To allow them all, run
`/permissions` and add `mcp__plugin_optlens_optlens` to the allowed tools. That allows every tool of the plugin's
server, including `run_python`, which runs the agent's code. To keep `run_python` asking, allow the other tools by
name instead, such as `mcp__plugin_optlens_optlens__open_model`.

The server (`optlens-mcp`) must be on the PATH Claude Code sees. On macOS, keep the venv out of folders that iCloud
syncs, such as `~/Documents` and `~/Desktop`. iCloud flags files hidden, and Python 3.13 skips hidden `.pth` files.
Then an editable install's `optlens-mcp` cannot import the package. Running `chflags nohidden` on the `.pth` files
fixes it. `OPTLENS_SOLVER` picks the solver ([Solvers](../README.md#solvers)).

The plugin adds two things to Claude Code:
- **The optlens tools.** These include `add_model` and `compare_models` for several models or scenarios side by side,
  and `save_model_context` for one saved description of each model (`.optlens/context/`). They also include
  `run_python`, a persistent Python process with the engine preloaded as `session`. It loads the model once, runs
  several steps per call, and shares its versions with the other tools.
  Solves stop at 45 s unless the agent asks for more, up to 95 s, when a solve was still improving. The plugin sets
  `OPTLENS_CALL_LIMIT` to 110. That is under 2 minutes, the point past which Claude Code moves a call to the
  background. The plugin also sets its server's `timeout` to 10 minutes, which overrides a shorter
  `MCP_TOOL_TIMEOUT` (some environments set 60 s). To change the limit, set `OPTLENS_CALL_LIMIT` (seconds, 30 to 590)
  before starting `claude`. `open_model` states the limits in force, and results say when a limit cut them short.
- **The `optlens` skill.** It says how to diagnose, explain and compare, in the same text as the server's
  instructions.

**Security.** `run_python` executes the code the agent writes, with your permissions. Its process starts without your
API keys and tokens, and Claude Code asks before each call. Opening a `.py` model runs that file up to its first solve
call. Open only models and code you trust. See [SECURITY.md](../SECURITY.md).

## Windows

The engine and the plugin work on Windows, tested on Python 3.12 and 3.13. `scripts/sandbox.sh` needs bash. In
PowerShell, these steps make the same clean setup. `$R` is the optlens checkout, or its folder from a zip. `$T` is the
new folder.

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

To start it, and again for each later session in a new PowerShell, do three things. Set the same
`CLAUDE_CONFIG_DIR`. Put the venv first on the PATH, so Claude Code finds `optlens-mcp`. Then start `claude` in the
project:

```powershell
$T = "$HOME\optlens-try"; $env:CLAUDE_CONFIG_DIR = "$T\claude-config"; $env:PATH = "$T\venv\Scripts;$env:PATH"
Set-Location "$T\project"; claude                                     # claude --continue resumes
```

The questions are in `packs/hotel/examples/try_week/questions.md`. A Python script that uses the engine needs the
`if __name__ == "__main__":` guard, because Windows starts solver processes with `spawn`. See the Quickstart in the
README.
