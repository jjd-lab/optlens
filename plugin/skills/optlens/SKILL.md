---
name: optlens
description: Debug and explain an LP/MILP optimization model with the optlens tools - why it is infeasible and what fixes it, what-if and why-not questions, what a limit is worth. Use when the user has a model file (.lp, .mps), Pyomo, gurobipy or PuLP code that builds a model, or asks about a solver result.
---

The optlens MCP server sends the full method as its instructions when it connects: follow them. In short, open the
model with open_model, save its business context once if it has none, answer a multi-step question with one
run_python call, lead with the cause in the model's business terms, and take every number from a tool result.

If the optlens tools are missing, the engine is not installed where the client starts it: tell the user to run
pip install "optlens[scip,mcp]" in that Python environment and restart the client.
