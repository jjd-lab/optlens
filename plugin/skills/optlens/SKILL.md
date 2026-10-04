---
name: optlens
description: Debug and explain an LP/MILP optimization model with the optlens tools - why it is infeasible and what fixes it, what-if and why-not questions, what a limit is worth. Use when the user has a model file (.lp, .mps), Pyomo, gurobipy or PuLP code that builds a model, or asks about a solver result.
---

# Debugging and explaining optimization models with optlens

Work through the optlens tools, not by reading the model file: LP and MPS files of real models run to megabytes, and
the tools give the same facts in a few lines. Open the model first: `open_model(path)` (add `document` if a file describes the model). The path can be an .lp or
.mps file, or the Python file that builds a Pyomo, gurobipy or PuLP model (`model.py`, or `model.py:name` for a
named model or a no-argument builder function); a Python file is run up to its first solve call, so open only the
user's own code. The answer says which
solvers are installed and which one will run.

## Model context (the shared vocabulary)
`open_model` shows the model's saved context: what each constraint and variable family means, what its indices are,
and the documented result. Use those meanings and names in every answer, so the same model is described the same way
in every session and by everyone on the team. If it says there is no saved context, write it once before answering:
read the document (or the model's code) and call `save_model_context`, mapping every listed family with its exact name;
if the reply lists families still undescribed, call it again with them added. The context is saved in
`.optlens/context/` in the project, as JSON the user can review, correct and commit; a new document or a changed set
of families asks for a new one.

## Answer shape
Lead with the cause in the model's business terms, then the evidence, then the options with their verified effect.
A planner reads it, not a solver developer.

## Infeasible models
- An IIS (`compute_iis`) is a minimal conflicting subset: a proof, not a census. Trace linking rows through to the
  business rules and limits that actually clash.
- A limit that breaks the pattern of its siblings is likely a data error (`suspicious_values`): say so, show the
  pattern, and verify that the typical value solves. Otherwise treat every requirement as intended.
- Offer every lever that fixes the model on its own, on both sides of the conflict, with the exact amount from a
  tool (`fix_menu`, `attainable_limit`, `feasibility_relaxation`), never from hand arithmetic.
- Verify each fix you recommend by re-solving (`modify_and_resolve`, or `try_options` for several) and report the
  actual status.

## Working models
- What-if: change and re-solve; why-not: `why_not`; the value of a limit: `marginal_value` (both directions; they
  differ at a kink, report both).
- Before saying how the plan changed, `compare_versions`; describe only changes you have seen.
- Every number you state comes from a tool result or is simple arithmetic on one, written out.

## Several models (scenarios)
When a question compares models (scenarios of one study, before/after files, alternatives), open the first with
`open_model` and the others with `add_model(path, name)`; every tool then takes the name as its version. Start with
`compare_models`: one table, the same metrics for each model (status, objective, rows at a limit per family, variable
totals). Then drill into each with the usual tools, and use `compare_versions` for decisions when the models share
their variables. Report every model the question names, with the same metrics for each, side by side.

Answer directly by default; subagents are optional. One conversation with `compare_models` and a few drill-down
calls handles most comparisons, and costs less. Hand models to `optlens:model-analyst` subagents only when the work
splits into independent, multi-step pieces: several models that each need their own diagnosis (an infeasible one to
explain, a driver to find), where running them in parallel saves time. One analyst per such model; none for a model
the comparison table already answers.

An analyst starts with a fresh context and sees only its brief, so the brief must carry everything it needs:
- **Objective**: the user's question in their words, why they ask it, and this analyst's part of it.
- **Context**: the model's name in the session (open every model first; analysts cannot open models), the business
  meanings it needs from the saved model context, and what you already found (the `compare_models` rows for its
  model), so it does not redo or contradict them.
- **Output**: the exact metrics, defined the same way in every analyst's brief, and anything extra you need back.
- **Boundaries**: what not to do (other models, re-solving what you already have) and roughly how deep to go.

Assemble the comparison yourself from the reports and your table; check any number that disagrees with a tool result
or with another report before you use it.

## Many solves
When a question needs several steps or many solves (a sweep, a search, a comparison of many limits, a check before
an answer), use `run_python` instead of calling tools one by one: `session` there has every tool as a method on the
open model, variables persist between calls, and the model is loaded once (a script run from the shell reloads it
every time, slow on large models). `session.try_options(...)` runs a batch. Print only what the answer needs.
Versions made in `run_python` and by the other tools are separate: compare within one of them.

## Solvers
Say which solver ran (the first result tells you). Don't ask the user to choose up front. Offer another solver only
when it matters: a solve hit its time limit, a large MIP and a faster solver is installed, two solvers disagree, or
the user must match a production solver. Gurobi is used only if the user has installed and licensed it; set
`OPTLENS_SOLVER` to highs, scip or gurobi to choose once.
