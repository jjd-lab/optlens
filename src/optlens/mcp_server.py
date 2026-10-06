"""MCP server over optlens.session: `open_model` loads an LP/MPS file into a session, and every tool in
`optlens.session.TOOLS` then works on it. Run as `optlens-mcp` (stdio); needs the `mcp` extra.

Solver choice: OPTLENS_SOLVER=highs|scip|gurobi prefers that solver; unset or "auto" prefers Gurobi when gurobipy
is installed (a user who licensed it likely runs it in production) and otherwise routes to HiGHS or SCIP per model.
A session on Gurobi does every step on Gurobi. Gurobi chosen by the user (OPTLENS_SOLVER or open_model's solver)
that cannot run a model stops and asks; under "auto" it falls back to HiGHS or SCIP routing, and the first result
says which solver ran and why. HiGHS and SCIP, both open source, may stand in for each other; results say so.

Model context: the business meaning of each family, written once by the host model from the document or code
(`save_model_context`) and kept in .optlens/context/ (or OPTLENS_CONTEXT_DIR), keyed by document and family
structure; `open_model` shows it in the same words every time, so sessions and people use one vocabulary.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import threading
import time
from pathlib import Path

import anyio
from jsonschema import Draft202012Validator
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult, TextContent, Tool

import optlens as od
from optlens import prompts
from optlens.context import ContextStore, _inventory_text, context_key, inventory, render, validate
from optlens.session import MULTI_MODEL_TOOLS, TOOLS, Session, Version, chosen_solver
from optlens.structure import names_are_meaningful, row_classes
from optlens.workspace import CodeWorkspace, tool_description

OPEN_MODEL = {
    "name": "open_model",
    "description": ("Load an LP or MPS model file (optionally .gz), or a Python file that builds a Pyomo, gurobipy or "
                    "PuLP model (model.py, or model.py:name for a model or a no-argument builder function; the file is "
                    "run, and stops at its first solve call, whose model is taken), "
                    "and make it the current model; its original is "
                    "version 'v0'. Every other tool works on the current model. Optional document: a text file that "
                    "describes the model (read with read_model_document). Optional solver: highs, scip or gurobi, "
                    "to prefer over the default choice."),
    "input_schema": {"type": "object", "properties": {
        "path": {"type": "string", "description": "Path to the .lp, .mps or .py file."},
        "document": {"type": "string", "description": "Path to a document describing the model, optional."},
        "solver": {"type": "string", "enum": ["highs", "scip", "gurobi"]}},
        "required": ["path"]},
}

_ITEM = {"type": "object", "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
         "required": ["name", "description"]}
SAVE_MODEL_CONTEXT = {
    "name": "save_model_context",
    "description": ("Save the current model's business context once, from its document or code, so every later "
                    "session shows the same meanings (the project's standard vocabulary for this model). Map EVERY "
                    "family open_model listed, with its exact name; do not describe families that are not listed (put "
                    "components the code defines but the model lacks in not_in_model). Explain each index position in "
                    "order. Mark linking=true for balance, flow, linking and definitional families. Copy numbers "
                    "exactly. Returns the coverage (families left undescribed) and the context as it will be shown."),
    "input_schema": {"type": "object", "properties": {
        "overview": {"type": "string", "description": "3-5 sentence business overview"},
        "objective": {"type": "string", "description": "direction and what is optimized, in business terms"},
        "documented_result": {"type": "string", "description": "the solved result the document reports (objective "
                              "and headline decisions), numbers copied exactly; empty if none"},
        "indices": {"type": "array", "items": _ITEM, "description": "index sets and what they mean"},
        "input_data": {"type": "array", "items": _ITEM, "description": "input data and what it means"},
        "families": {"type": "array", "items": {"type": "object", "properties": {
            "family": {"type": "string", "description": "exact family name from open_model's list"},
            "kind": {"type": "string", "enum": ["constraint", "variable"]},
            "meaning": {"type": "string", "description": "business meaning of one member, one or two sentences"},
            "index_meaning": {"type": "string", "description": "what each index position means, in order"},
            "linking": {"type": "boolean"}},
            "required": ["family", "kind", "meaning"]}},
        "not_in_model": {"type": "array", "items": {"type": "string"}}},
        "required": ["overview", "objective", "families"]},
}


DEFAULT_CALL_LIMIT = 60.0  # the per-call timeout of many MCP clients (Claude Desktop, the TypeScript SDK)
MIN_CALL_LIMIT = 30.0
DEFAULT_SOLVE, DEFAULT_IIS = 45.0, 40.0  # a solve and a large MIP's IIS search unless a call asks for more


def call_limits(value: str | None = None) -> tuple[float, float, float, float]:
    """(run_python's call timeout, the default solve limit, a large MIP's default IIS budget, the most a tool's
    time_limit may ask for) for a tool call that must answer within OPTLENS_CALL_LIMIT seconds (``value``, else the
    environment; default 60, at least 30). The defaults stay at 45 and 40 s: a call answers in under a minute, which
    every client accepts and Claude Code does not move to the background (it does past 120 s); an agent asks for more
    only when the solve showed progress."""
    raw = os.environ.get("OPTLENS_CALL_LIMIT", "") if value is None else value
    try:
        limit = float(raw) if raw.strip() else DEFAULT_CALL_LIMIT
    except ValueError:
        limit = DEFAULT_CALL_LIMIT
    limit = max(MIN_CALL_LIMIT, limit) if limit == limit else DEFAULT_CALL_LIMIT  # NaN reads as unset
    return limit - 10.0, min(DEFAULT_SOLVE, limit - 15.0), min(DEFAULT_IIS, limit - 20.0), limit - 15.0


# per run_python call; per solve by default; a large MIP's IIS search (the result says when it stopped short:
# "reduced"); the most a tool's time_limit may ask for
RUN_PYTHON_LIMIT, CALL_SOLVE_LIMIT, CALL_IIS_BUDGET, CALL_MAX_SOLVE = call_limits()
CALL_LIMIT = RUN_PYTHON_LIMIT + 10.0
LONGER = ("modify_and_resolve",)  # tools whose time_limit only this server offers (the bench's agents do not see it)


def limits_line() -> str:
    """The time limits as open_model states them, so the agent (and the user it tells) knows them from the start."""
    return (f"Time limits: solves stop at {CALL_SOLVE_LIMIT:.0f} s unless a tool's time_limit asks for more, up to "
            f"{CALL_MAX_SOLVE:.0f} s; each tool call answers within {CALL_LIMIT:.0f} s (OPTLENS_CALL_LIMIT), "
            f"run_python calls within {RUN_PYTHON_LIMIT:.0f} s. A result cut short by a limit says so. If a call times "
            "out in the client, the user should lower OPTLENS_CALL_LIMIT to the client's limit and restart the server.")


def with_limits(tool: dict) -> dict:
    """A tool's schema with its time_limit text saying this server's limits (the session's own say 60 s), and a
    time_limit added to the tools in LONGER."""
    props = dict(tool["input_schema"].get("properties", {}))
    if tool["name"] in LONGER:
        props["time_limit"] = {"type": "number", "description": "Seconds for the solve; default 60."}
    if "time_limit" not in props:
        return tool
    text = props["time_limit"]["description"]
    most = f"at most {CALL_MAX_SOLVE:.0f} here. Ask for more only when an earlier solve of this model stopped at its " \
           "limit while the solver was still improving."
    text = text.replace("default 60.", f"default {CALL_SOLVE_LIMIT:.0f}, {most}")
    text = text.replace("default 300, or 60 on a MIP over 500 rows.",
                        f"default {CALL_SOLVE_LIMIT:.0f}, or {CALL_IIS_BUDGET:.0f} on a MIP over 500 rows; {most}")
    schema = {**tool["input_schema"], "properties": {**props, "time_limit": {**props["time_limit"], "description": text}}}
    return {**tool, "input_schema": schema}


RUN_PYTHON = {
    "name": "run_python",
    "description": tool_description(RUN_PYTHON_LIMIT, note=(
        "It opens the model open_model opened and shares its versions with the other tools: a version either "
        "made is in both, with its solve. After a time-out the process restarts and its variables are gone. ")),
    "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
}


def available_solvers() -> list[str]:
    return ["highs"] + [name for name, mod in (("scip", "pyscipopt"), ("gurobi", "gurobipy"))
                        if importlib.util.find_spec(mod)]


def preferred_solver(explicit: str | None = None) -> tuple[str | None, bool]:
    """(the solver to prefer, whether the user chose it): their choice, else Gurobi when gurobipy is installed."""
    if choice := chosen_solver(explicit):
        return choice, True
    return ("gurobi" if "gurobi" in available_solvers() else None), False


class State:
    def __init__(self, store: ContextStore | None = None):
        self.session: Session | None = None
        self.name = ""
        self.store = store
        self.md = self.inv = self.key = None
        self.source = ""
        self.model_spec, self.doc = "", None  # what open_model loaded, for the run_python workspace
        self.workspace: CodeWorkspace | None = None  # started on the first run_python call, one per open model
        self.ws_lock = threading.Lock()  # one snippet at a time: the workspace is one process
        # A client may call tools in parallel. Calls that replace or add models take this lock; the others run in
        # parallel (the session allocates version ids under its own lock, and solves run in separate processes), so
        # one long search does not make another call time out.
        self.lock = threading.Lock()

    def _context(self, interp: dict | None) -> str:
        coverage = validate(interp, self.inv)[1] if interp else None
        meaningful = names_are_meaningful(self.md)
        shapes = [] if meaningful else [(label, len(idx), self.md.row_names[idx[0]])
                                        for label, idx in list(row_classes(self.md).items())[:40]]
        return render({"inventory": self.inv, "interpretation": interp, "coverage": coverage,
                       "names_meaningful": meaningful, "shapes": shapes})

    def open_model(self, path: str, document: str | None = None, solver: str | None = None) -> str:
        file, sep, attr = path.rpartition(":") if re.search(r"\.py:[A-Za-z_]\w*$", path) else (path, "", "")
        p = Path(file).expanduser().resolve()
        md = od.load(f"{p}{sep}{attr}")
        # the document, else a Python model's own code, is what the context is written from
        doc = Path(document).expanduser().resolve() if document else p if p.suffix == ".py" else None
        # many MCP clients stop waiting for a tool call after 60 s (OPTLENS_CALL_LIMIT): solves and large searches
        # stop in time to answer
        prefer, only = preferred_solver(solver)
        self.session = Session({"v0": Version(md, None, "original model")}, str(doc) if doc else None,
                               prefer=prefer, only_prefer=only, time_limit=CALL_SOLVE_LIMIT,
                               large_mip_iis_budget=CALL_IIS_BUDGET, max_time_limit=CALL_MAX_SOLVE)
        self.name = p.name
        self.md, self.inv = md, inventory(md)
        self.model_spec, self.doc = f"{p}{sep}{attr}", doc
        self.close_workspace()  # its model is the previous one
        self.source = doc.name if doc else ""
        self.key = context_key(self.inv, doc.read_text(errors="replace") if doc else "")
        self.store = self.store or ContextStore()
        interp = self.store.load(md.name, self.key)
        if interp is not None:
            context = f"Saved model context ({self.store.path(md.name, self.key).name}):\n{self._context(interp)}"
        elif doc is not None and names_are_meaningful(md):
            context = (f"No saved model context for this model and {doc.name}. Before answering, read the document "
                       "(read_model_document) and call save_model_context once, mapping every family below; later "
                       f"sessions will show it in the same words.\nFamilies to map:\n{_inventory_text(self.inv)}")
        else:
            context = self._context(None)
        return (f"opened {p.name}: {md.num_rows} rows, {md.num_cols} columns"
                f"{' (' + str(int(md.is_int.sum())) + ' integer)' if md.is_mip else ''}; solvers installed: "
                f"{', '.join(available_solvers())}; preferred: {self.session.prefer or 'none (routed per model)'}\n"
                + (f"This session uses only {prefer}, the user's choice: in run_python solve with "
                   f"`od.BACKENDS['{prefer}']`, never another solver.\n" if self.session.strict() else "")
                + self.session.get_model_overview() + "\n" + limits_line() + "\n\n" + context)

    def save_model_context(self, **interp) -> str:
        clean, coverage = validate(interp, self.inv)
        path = self.store.save(self.md.name, self.key, clean, self.source)
        missing = coverage["undescribed"]
        return (f"saved to {path}" + (f"; still undescribed: {', '.join(missing)} (call again with them added)"
                                      if missing else "; every family is described")
                + (f"; not in this model, left out: {', '.join(coverage['unknown'])}" if coverage["unknown"] else "")
                + "\n" + self._context(clean))

    def run_python(self, code: str) -> tuple[str, bool]:
        with self.ws_lock:
            if self.workspace is None:
                import tempfile

                v0 = self.session.versions["v0"].result  # open_model solved it: the workspace starts from it
                self.workspace = CodeWorkspace(self.model_spec, Path(tempfile.mkdtemp(prefix="optlens-ws-")),
                                               str(self.doc) if self.doc else None, timeout=RUN_PYTHON_LIMIT,
                                               solve_limit=CALL_SOLVE_LIMIT, prefer=self.session.prefer,
                                               only_prefer=self.session.only_prefer,
                                               base=(self.session.route, v0) if v0 is not None else None)
            t0 = time.time()
            out, err = self.workspace.run_python(code, self.session)
            return f"{out}\n[{time.time() - t0:.1f} s]", err

    def close_workspace(self) -> None:
        if self.workspace is not None:
            self.workspace.close()
            self.workspace = None

    def call(self, name: str, args: dict) -> tuple[str, bool]:
        if name in ("open_model", "add_model", "save_model_context"):
            with self.lock:
                return self._call(name, args)
        return self._call(name, args)

    def _call(self, name: str, args: dict) -> tuple[str, bool]:
        if name != "open_model" and self.session is None:
            return "no model is open: call open_model with the path of an .lp, .mps or .py file first", True
        if name == "run_python":
            try:
                return self.run_python(str(args.get("code", "")))
            except Exception as e:  # the workspace could not start (it says where its log is)
                return f"{type(e).__name__}: {e}", True
        if name not in ("open_model", "save_model_context"):
            return self.session.call(name, args)
        t0 = time.time()
        try:
            out = getattr(self, name)(**args)
        except Exception as e:  # tool errors go back to the model as error results
            return f"{type(e).__name__}: {e}", True
        note = ""
        if self.session is not None:
            note, self.session.route_note = self.session.route_note, ""
        return f"{out}\n[{time.time() - t0:.1f} s]" + (f"\n[{note}]" if note else ""), False


def argument_problems(schema: dict, args: dict) -> str:
    """Why args do not fit a tool's schema, with the schema itself, or "" when they fit. A client can defer a
    server's tools (Claude Code's tool search), so an agent may call a tool whose schema it never saw."""
    unknown = sorted(set(args) - set(schema.get("properties", {})))
    found = [f"unknown argument {k!r}" for k in unknown]
    found += [f"{'.'.join(map(str, e.absolute_path)) or 'arguments'}: {e.message}"
              for e in Draft202012Validator(schema).iter_errors(args)]
    if not found:
        return ""
    return ("the arguments do not match this tool's input schema; nothing was run:\n"
            + "\n".join(f"- {f}" for f in found[:5]) + f"\ninput schema: {json.dumps(schema)}")


def build_server(state: State | None = None) -> Server:
    state = state or State()
    tools = [Tool(name=t["name"], description=t["description"], input_schema=t["input_schema"])
             for t in map(with_limits, [OPEN_MODEL, SAVE_MODEL_CONTEXT, *MULTI_MODEL_TOOLS, *TOOLS, RUN_PYTHON])]
    schemas = {t.name: t.input_schema for t in tools}

    async def list_tools(ctx, params) -> ListToolsResult:
        return ListToolsResult(tools=tools)

    async def call_tool(ctx, params: CallToolRequestParams) -> CallToolResult:
        if params.name not in schemas:
            return CallToolResult(content=[TextContent(type="text", text=f"unknown tool {params.name}")], is_error=True)
        args = dict(params.arguments or {})
        if problems := argument_problems(schemas[params.name], args):
            return CallToolResult(content=[TextContent(type="text", text=problems)], is_error=True)
        # solves run in worker processes; keep the event loop free while waiting for them
        out, err = await anyio.to_thread.run_sync(state.call, params.name, args)
        return CallToolResult(content=[TextContent(type="text", text=out)], is_error=err)

    return Server("optlens", instructions=prompts.SERVER_INSTRUCTIONS,
                  on_list_tools=list_tools, on_call_tool=call_tool)


async def _serve() -> None:
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    anyio.run(_serve)


if __name__ == "__main__":
    main()
