---
name: optlens
description: Debug and explain an LP/MILP optimization model with the optlens tools - why it is infeasible and what fixes it, what-if and why-not questions, what a limit is worth. Use when the user has a model file (.lp, .mps), Pyomo, gurobipy or PuLP code that builds a model, or asks about a solver result.
---

Debug and explain LP/MILP optimization models: why a model is infeasible and what fixes it, what-if and why-not
questions, what a limit is worth, how plans or scenarios differ.

Work through the tools, not by reading the model file: real LP and MPS files run to megabytes, and the tools give the
same facts in a few lines. Open the model first with open_model(path), adding document if a file describes the model.
The path can be an .lp or .mps file, or the Python file that builds a Pyomo, gurobipy or PuLP model (model.py, or
model.py:name for a named model or a no-argument builder); a Python file is run up to its first solve call, so open
only the user's own code. The answer says which solvers are installed and which one will run. open_model reads .lp,
.mps and .gz files itself; read a document such as model.md with the client's file-reading tool (Read in Claude Code),
not by shelling out (cat, gzip -dc), which asks for permission and depends on the shell (Git Bash on Windows).

Resumed conversations: each one starts a new server. open_model on the same model brings back the versions an earlier
session made ("restored N versions"): when it does, tell the user, and check version_history before saying something
was not done earlier. A tool that answers "no model is open" means the server restarted: call open_model on the same
file, which restores the earlier versions.

Load the tools before using them: a client may list them as deferred, with names only. Load them by the names the
client lists, which carry a prefix (in Claude Code mcp__plugin_optlens_optlens__open_model, or mcp__optlens__open_model
when the server was added by hand), at least open_model, the tools you plan to call and run_python, whose description
lists every engine method with its arguments and what a solve returns. Then call them as described instead of
probing the API. Every tool answers within the server's time limit (open_model states it); if the client stops a call
first ("timed out"), the client waits less than OPTLENS_CALL_LIMIT: tell the user to set OPTLENS_CALL_LIMIT to the
client's limit (60 for most clients) and restart the server, rather than solving the model another way.

Model context (the shared vocabulary): open_model shows the model's saved context, what each constraint and variable
family means and the documented result. Use those meanings and names in every answer. If it says there is no saved
context, write it once before answering: read the document (or the model's code) and call save_model_context, mapping
every listed family with its exact name; if the reply lists families still undescribed, call it again with them. The
context is saved as JSON in .optlens/context/ in the project, for the user to review, correct and commit.

Answer shape: lead with the cause in the model's business terms, then the evidence, then the options with their
verified effect. A planner reads it, not a solver developer. Every number you state comes from a tool result or is
simple arithmetic on one, written out.

Infeasible models: an IIS (compute_iis) is a minimal conflicting subset, a proof, not a census; trace linking rows
through to the business rules and limits that actually clash. A limit that breaks the pattern of its siblings is
likely a data error (suspicious_values): say so, show the pattern, and verify that the typical value solves;
otherwise treat every requirement as intended. Offer every lever that fixes the model on its own, on both sides of
the conflict, with the exact amount from a tool (fix_menu, attainable_limit, feasibility_relaxation), never from hand
arithmetic. Verify each fix you recommend by re-solving (modify_and_resolve, or try_options for several) and report
the actual status.

Working models: what-if is change and re-solve (modify_and_resolve); why-not is why_not; the value of a limit is
marginal_value (both directions; when they differ the optimum sits at a kink, report both). Before saying how the
plan changed, compare_versions; describe only changes you have seen.

Several models (scenarios, before/after, alternatives): open the first with open_model and the others with
add_model(path, name); every tool then takes the name as its version. Start with compare_models (the same metrics for
each model side by side), then drill into each; compare_versions works across models that share their variables.
Report every model the question names, with the same metrics for each.

Many solves: when a question needs several steps or many solves (a sweep, a search, a check before an answer), use
run_python instead of calling tools one by one: session there has every tool as a method on the open model,
variables persist between calls, and the model is loaded once. Print only what the answer needs. Versions made in
run_python and by the other tools are shared, with their solves.

Large MIPs and time: open_model's base solve is the probe. Proven optimal: the defaults are fine. Stopped at its
limit: report the plan with its bound and gap, and read the note on its starting plan. If the solver improved on the
start, a longer time_limit on the what-ifs that matter (modify_and_resolve, up to the most open_model states) can
narrow the gap; if it found no better plan, more time is unlikely to help, so report each effect as the range its gaps
allow. Never narrow that range by judgement: an LP relaxation's change bounds nothing for the MIP; give it, if at all,
as an indication, beside the proven range. A call that runs for minutes may be moved to the background by the client:
wait for its result, and never start another solve of the same model meanwhile (it competes for the same processors).

Solvers: say which solver ran. Don't ask the user to choose up front; offer another solver only when it matters (a
solve hit its time limit, two solvers disagree, the user must match a production solver). Gurobi is used only if the
user has installed and licensed it; OPTLENS_SOLVER=highs|scip|gurobi chooses once.
