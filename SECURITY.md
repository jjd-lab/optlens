# Security

## Reporting a vulnerability

Email **hello@optlens.dev** with "security" in the subject: what you found, how to reproduce it, and the optlens
version (`python -c "import optlens; print(optlens.__version__)"`). Please do not open a public issue for a
vulnerability. You will get a reply within a week.

## What optlens does on your machine

optlens is a local tool with no network service and no sandbox:
- Every solve runs in a worker process with a hard time limit.
- `run_python` (the Claude Code plugin's code tool, `optlens.workspace`) executes the code an agent writes, with your
  user's permissions. Its process starts with a short allow-list of environment variables, so API keys and tokens in
  your environment are not passed on, but the code can read and write whatever your user can.
- Loading a `.py` model (`optlens.load("model.py")`, the MCP server's `open_model`) runs that file up to its first
  solve call, the way `python model.py` would.

Open only models and code you trust, as you would before running them yourself.
