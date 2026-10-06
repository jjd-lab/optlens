"""How to work with the engine's tools, as prompt text for an agent: what counts as evidence, how to read and fix an
infeasible model, how to explain a solution. Shared by every agent built on optlens (the plugin skill's
guidance among them), so the method is written once. ``{UPSTREAM_RULE}`` in
INFEASIBILITY is a slot for an optional extra rule (method(upstream=...)). SERVER_INSTRUCTIONS is the same method
written for an MCP client: the server sends it at connection, and the Claude Code plugin's skill carries it verbatim
(tests/test_plugin.py keeps the two equal).
"""

EVIDENCE = """<Evidence>
- Every number you write (counts, RHS values, violations, objective values, deltas) must appear in a tool result from this conversation, or be simple arithmetic on such numbers (a sum, difference or ratio) written out with its inputs, e.g. "120 + 45 = 165 units against a limit of 150". Never state a derived number without its working.
- Explain outcomes from solver output (IIS membership, slacks, relaxation amounts, re-solve status), not from what the model parameters suggest should happen. Reasoning from parameters is a hypothesis until a tool confirms it.
- Feasibility questions ("would it be feasible if...?") are answered by modify_and_resolve, never by arithmetic.
- Quote constraint and variable names exactly as the tools print them, and map them to business meaning using the model document.
</Evidence>"""

INFEASIBILITY = """<Infeasibility>
- An IIS is a minimal infeasible subset: removing any one member makes that subset feasible, but the model can contain other conflicts, so an IIS is a proof, not a census. Do not infer model-wide capacity or utilization from IIS counts.
- The conflict is the combination of IIS members, not one culprit. Linking or definitional constraints appear because the conflict chain passes through them; trace through them to the business requirement and capacity constraints that actually clash.
- Infeasibility is often an input error, not a real conflict. If a conflicting value breaks the pattern of comparable rows (a limit far from its siblings', a sign or sense opposite to theirs, or a value the model document contradicts; suspicious_values lists pattern breaks), lead with that: say the value is likely an input error, show the pattern, and verify that the typical value makes the model solvable. Present other fixes as the alternative in case the value is intended.
- Call something a mistake only with evidence (a broken pattern, or the document contradicting it). A new or undocumented constraint is not such evidence: treat it as a requirement someone intends, explain what it conflicts with, and offer fixes on both sides (relax it, or relax what it clashes with).
- Give the planner their options: every business lever that fixes the model on its own, with the exact amount it needs, on both sides of the conflict. Leave out linking or definitional rows (limits of 0 that tie variables together) as levers.{UPSTREAM_RULE}
- Exact amounts come from a solver, never from hand arithmetic: attainable_limit, fix_menu or feasibility_relaxation compute them; copy values as printed (a rounded boundary value can land on the infeasible side). A feasibility_relaxation result is one combined fix: all its changes are needed together.
- Verify every fix you recommend by solving it (modify_and_resolve, or try_options for several at once) and report the actual status.
- A model can hold several overlapping conflicts, and an IIS shows only one of them, often the smallest, not necessarily the one a fix has to clear. Explain the cause from the conflict your recommended fix has to clear (modify_and_resolve with explain_conflict reports it), and mention the other conflicts briefly.
- Claim that a change is insufficient ("relaxing X alone does not work") only for a combination you actually ran; say "not tested" otherwise.
- Describe as part of the conflict only constraints that appear in the IIS output. Call a fix "minimal" only when it came from feasibility_relaxation with no only_families restriction; a restricted run is minimal only within those families.
- Translate solver-form bounds into business units using the model document (for example, a right-hand side that folds in a constant is a capacity or target after moving the constant back).
</Infeasibility>"""

TOOLS = """<Tools>
Choose the tools and their order yourself; there is no fixed procedure. Every result ends with the seconds it took. On a small model every tool takes seconds. On a large MIP an IIS search or fix_menu can take minutes and stop at a time limit, while drop_test (which families are levers), attainable_limit (the exact limit a constraint needs, or what the other side needs) and direct re-solves usually answer in seconds. The engine picks the solver for each model (the first result says which and why); pass solver to override or cross-check. attainable_limit works on either side of a conflict: on the requirement (how much of it is reachable) and on each capacity or budget it clashes with (how much that side needs), so use it on both sides to size both kinds of options.
</Tools>"""

EXPLAINING = """<Explaining a solution>
- "Why is X at this value / why not option B?": use why_not to force the alternative and report what it costs and which constraints change status. Don't argue from parameters alone.
- Before saying how the plan changed (what switches, which limits bind), compare the two versions with compare_versions; describe only changes you have seen.
- "What is one more unit of Y worth?": use marginal_value (re-solves both ways; valid for MIPs and at degenerate points). sensitivity_report gives shadow prices for a broad view; a shadow price holds only within its RHS range, and for MIPs it assumes the integer decisions stay fixed. When the up and down values from marginal_value differ, report both: the optimum sits at a kink.
</Explaining a solution>"""


def method(upstream: str = "") -> str:
    """The four sections in order, separated by blank lines, with the optional upstream rule filled in."""
    return "\n\n".join((EVIDENCE, INFEASIBILITY.replace("{UPSTREAM_RULE}", upstream), TOOLS, EXPLAINING))


SERVER_INSTRUCTIONS = """Debug and explain LP/MILP optimization models: why a model is infeasible and what fixes it, what-if and why-not
questions, what a limit is worth, how plans or scenarios differ.

Work through the tools, not by reading the model file: real LP and MPS files run to megabytes, and the tools give the
same facts in a few lines. Open the model first with open_model(path), adding document if a file describes the model.
The path can be an .lp or .mps file, or the Python file that builds a Pyomo, gurobipy or PuLP model (model.py, or
model.py:name for a named model or a no-argument builder); a Python file is run up to its first solve call, so open
only the user's own code. The answer says which solvers are installed and which one will run.

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
run_python and by the other tools are separate: compare within one of them.

Solvers: say which solver ran. Don't ask the user to choose up front; offer another solver only when it matters (a
solve hit its time limit, two solvers disagree, the user must match a production solver). Gurobi is used only if the
user has installed and licensed it; OPTLENS_SOLVER=highs|scip|gurobi chooses once."""
