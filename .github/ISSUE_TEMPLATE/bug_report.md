---
name: Wrong or unhelpful result
about: optlens crashed, gave a wrong answer, or missed the real cause on your model
labels: bug
---

**The model.** Attach the file (LP, MPS, or the Python that builds it), or a smaller one that shows the same problem.
If you cannot share it, describe its size and structure.

**What you ran.** The call or the question (Python, MCP tool, or Claude Code prompt).

**What optlens said**, and **what you expected**.

**Environment.** optlens version (`python -c "import optlens; print(optlens.__version__)"`), Python version, OS,
solver (`OPTLENS_SOLVER`, and the Gurobi license type if you use Gurobi).
