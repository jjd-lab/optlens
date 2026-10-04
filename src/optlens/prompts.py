"""How to work with the engine's tools, as prompt text for an agent: what counts as evidence, how to read and fix an
infeasible model, how to explain a solution. Shared by every agent built on optlens (the plugin skill's
guidance among them), so the method is written once. ``{UPSTREAM_RULE}`` in
INFEASIBILITY is a slot for an optional extra rule (method(upstream=...)).
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
