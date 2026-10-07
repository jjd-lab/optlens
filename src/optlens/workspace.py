"""A persistent Python workspace for an agent: one process per conversation that keeps its variables between calls,
with the engine preloaded as `session` (every tool, as a method) and `od` (optlens, for numbers inside loops). Used by
the plugin's `run_python` tool (optlens.mcp_server) and by any agent that writes code against the engine: as correct as
calling the tools one by one, at lower cost.

The parent talks to the worker over a local socket; the worker runs each snippet with exec() in one namespace and
returns what it printed. Agent code must not see API keys, so the worker starts with an environment allow-list. There
is no sandbox beyond that: the code can do what the user running it can (in Claude Code the user approves each call).

Design notes (after Anthropic, "Code execution with MCP"):
state persists like a notebook; output is capped with a note on what to do instead; every result ends with the
names in scope, and a NameError names the closest one, because a bare NameError reads as lost state.
"""
from __future__ import annotations

import os
import secrets
import subprocess
import sys
import threading
from multiprocessing.connection import Client, Listener
from pathlib import Path

import numpy as np

TIMEOUT = 180.0
SOLVE_MARGIN = 10.0  # seconds a call keeps past its session's solve limit: the solver's grace and the reply
# Within one call (T s): solves and searches get at most what is left until T - STEP_MARGIN (backends.fit); code still
# running at T - STOP_MARGIN is stopped and its variables kept; a worker silent at T + KILL_GRACE is restarted.
STEP_MARGIN = 8.0
STOP_MARGIN = 3.0
KILL_GRACE = 5.0
OUT_CAP = 10_000
PRELOADED = ("session", "od", "np", "MODEL_FILE", "MODEL_DOC", "TOOL_DOCS")


def tool_description(timeout: float = TIMEOUT, note: str = "") -> str:
    """The `run_python` tool's description: what is preloaded, and every engine method with its real signature. ``note``
    goes before the closing advice (the plugin says its versions are shared with its other tools')."""
    import optlens as od
    from optlens.session import TOOLS, Session, chosen_solver

    solver = chosen_solver()
    if solver in od.BACKENDS and od.BACKENDS[solver].licensed:
        note = (f"This session uses only {solver}, the user's choice: solve with `od.BACKENDS['{solver}']`, never "
                f"another solver. " + note)
    else:
        solver = "highs"
    return (
        "Run Python in a process that keeps its variables between calls, like a notebook (timeout "
        f"{timeout:.0f} s per call; output capped at {OUT_CAP:,} characters). Preloaded: `session`, the diagnostic "
        "engine on this model (version 'v0' is the original; each method returns text to print; changes create "
        "versions v1, v2, ...); `od`, the optlens module, for numbers inside loops: `md = session.get('v0').md`, "
        "`md.set_row_bounds(name, lo=None, hi=None)` and `md.set_col_bounds(name, lb=None, ub=None)` return a changed "
        "copy, `md.lp_relaxation()` drops integrality (`md` is frozen: never assign to it), "
        f"`od.BACKENDS['{solver}'].solve(md, time_limit=60.0, start=None)` returns `.status`, `.obj`, `.bound`, `.gap`, "
        "`.x`, `.row_dual`, `.reduced_cost`, and `md.row_names`, `md.col_names`, `md.row_lo`, `md.row_hi`, `md.obj` (costs), "
        "`md.is_int` describe it; a version `v = session.get('v1')` has `v.md` and `v.changes`, "
        "`session.solved('v1')` its result (the same fields; solved if it was not); `np`; `MODEL_FILE` and "
        "`MODEL_DOC`, the model and document paths. "
        "`od.BACKENDS[...].solve` solves from scratch; `session` methods start a MIP of 5,000+ rows from a plan built "
        "from its LP relaxation, far better on large MIPs, so solve versions of a large MIP through `session`. "
        "The call's time is shared by everything in it: each solve or search gets at most what is left (the output "
        "says when one got less than asked), and code still running at the limit is stopped with its variables kept. "
        + note + "Do several quick steps in one call and print only what you need; give each long solve or search "
        "(an IIS, a fix menu, a time_limit near the call's) its own call. Methods of `session` (full description: "
        "print(TOOL_DOCS['name'])):\n" + api_lines(TOOLS, Session))


def api_lines(tools: list[dict], session_cls: type) -> str:
    """One line per engine tool, as a `session` method: its real Python signature (the tool schema's order can
    differ, and a wrong order costs a turn) and the first sentence of its description."""
    import inspect

    lines = []
    for t in tools:
        params = list(inspect.signature(getattr(session_cls, t["name"])).parameters.values())[1:]
        args = ", ".join(p.name if p.default is p.empty else f"{p.name}={p.default!r}" for p in params)
        first = t["description"].split(". ")[0].rstrip(".")
        lines.append(f"- session.{t['name']}({args}) -> str: {first}.")
    return "\n".join(lines)


class CodeWorkspace:
    """Parent side: starts the worker, sends snippets, restarts it after a timeout."""

    def __init__(self, model_path: str, workdir: Path, doc_path: str | None = None, timeout: float | None = None,
                 confine: list[str] | None = None, prefer: str | None = None, only_prefer: bool = False,
                 solve_limit: float | None = None, base=None):
        self.model_file = str(Path(model_path).resolve())
        self.doc_file = str(Path(doc_path).resolve()) if doc_path else ""
        self.timeout = timeout  # per call; None: TIMEOUT (read at call time)
        self.solve_limit = solve_limit  # the session's per-solve limit; None: TIME_LIMIT. Kept inside the call's timeout
        # (solver route, SolveResult) of the original model, already solved by the caller (the MCP server): the worker
        # starts with it instead of solving it again, so a call's first what-if gets the whole time limit
        self.base = base
        # confine: the code may read and write only inside the working directory, these extra paths (a pack, the
        # model and document) and Python's own installation, and may not start other programs (for testing an agent,
        # which must not find tools or answers elsewhere on disk). None: no limits, as for a user's own machine.
        self.confine = None if confine is None else [str(Path(p).resolve()) for p in confine]
        # the worker's Session gets these; without them it takes the user's choice from OPTLENS_SOLVER, if any
        self.prefer, self.only_prefer = prefer, only_prefer
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.proc = self.conn = None
        self.last_engine_calls: list[dict] = []  # the session methods the last snippet called, for the attribution check
        self._sent: set[str] = set()  # versions whose solve the worker has (run_python with a session)
        self._starting = False  # a restarted worker still loading the model: the next call waits for it
        self._start()
        self._await_ready()

    def _await_ready(self) -> None:
        if not self.conn.recv().get("ready"):
            raise RuntimeError(f"code worker failed to start; see {self.workdir / 'worker.log'}")
        self._starting = False

    def _start(self) -> None:
        from optlens.session import TIME_LIMIT

        key = secrets.token_bytes(16)
        listener = Listener(authkey=key)  # a Unix socket, or a named pipe on Windows
        keep = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "GRB_LICENSE_FILE", "OPTLENS_SOLVER", "PYTHONPATH",
                "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE")  # the last four: a Windows Python needs them to start
        env = {k: os.environ[k] for k in keep if k in os.environ}
        # the worker imports the parent's optlens, also when the install's .pth file is not read (macOS hidden flag)
        env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]),
                                             *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
        if self.confine is not None:
            env["CODE_WS_CONFINE"] = os.pathsep.join([str(self.workdir.resolve()), self.model_file,
                                                      *([self.doc_file] if self.doc_file else []), *self.confine])
        env.update(CODE_WS_TIMEOUT=str(self.timeout or TIMEOUT))
        env.update(MODEL_FILE=self.model_file, MODEL_DOC=self.doc_file, CODE_WS_ADDR=listener.address,
                   CODE_WS_KEY=key.hex(), CODE_WS_DOC=self.doc_file,
                   CODE_WS_SOLVE_LIMIT=str(max(1.0, min(self.solve_limit or TIME_LIMIT,
                                                        (self.timeout or TIMEOUT) - SOLVE_MARGIN))))
        if self.solve_limit is not None:  # a caller with a client timeout (the MCP server): a tool's time_limit may ask
            # for more than the default, within the call's timeout less the margin and the edits (6,240 set_rhs changes
            # and their version took 5 s besides a 90 s solve in a 100 s call)
            env["CODE_WS_MAX_LIMIT"] = str(max(1.0, (self.timeout or TIMEOUT) - SOLVE_MARGIN - 5.0))
        path = self.workdir / "v0_result.npz"
        if self.base is not None:
            route, r = self.base
            _save_result(path, r, route)
        if path.is_file():  # the caller's solve of the original model, or the one a worker before a restart saved
            route_file = path.with_suffix(".route")
            env.update(CODE_WS_V0=str(path.resolve()), CODE_WS_V0_SAVED="1",
                       CODE_WS_ROUTE=route_file.read_text() if route_file.is_file() else "")
        if self.prefer:
            env.update(CODE_WS_PREFER=self.prefer, CODE_WS_ONLY_PREFER="1" if self.only_prefer else "")
        with open(self.workdir / "worker.log", "a") as log:
            self.proc = subprocess.Popen([sys.executable, "-m", "optlens.workspace"], cwd=self.workdir, env=env,
                                         stdout=log, stderr=log)
        timer = threading.Timer(60, listener.close)  # a worker that dies before connecting must not hang the caller
        timer.start()
        try:
            self.conn = listener.accept()
        except OSError:
            raise RuntimeError(f"code worker did not start; see {self.workdir / 'worker.log'}") from None
        finally:
            timer.cancel()
            listener.close()
        self._starting = True

    def run_python(self, code: str, session=None) -> tuple[str, bool]:
        """Runs ``code``. With ``session`` (the MCP server's), the worker's session and it share their versions: the
        server's versions and solves go in first, and the versions and solves the code made come back."""
        msg = {"code": code}
        if session is not None:
            msg["versions"] = session.export_versions()
            msg["results"] = {vid: v.result for vid, v in session.versions.items()
                              if v.result is not None and vid not in self._sent}
            self._sent |= set(msg["results"])
        limit = self.timeout or TIMEOUT
        msg["timeout"] = limit
        try:
            if self._starting:  # restarted after the last call: it loaded the model meanwhile, or is finishing
                self._await_ready()
            self.conn.send(msg)
            if not self.conn.poll(limit + KILL_GRACE):  # the worker stops its own code at the limit; this is a hang
                return self._restart(f"timed out after {limit:.0f} s"), True
            r = self.conn.recv()
        except (EOFError, OSError):  # the worker died (out of memory on an 837k-row model): restart it, like a time-out
            return self._restart("the Python process stopped (it ran out of memory)"), True
        self.last_engine_calls = r["engine_calls"]
        if session is not None and r.get("versions"):
            session.restore_versions(r["versions"])
            for vid, res in r["results"].items():
                if vid in session.versions and session.versions[vid].result is None:
                    session.versions[vid].result = res
            self._sent |= set(r["results"])
        return r["out"], r["error"]

    def _restart(self, why: str) -> str:
        """A new worker, loading the model while the caller reads this answer; the next call waits for it."""
        self.close()
        self._start()
        self.last_engine_calls, self._sent = [], set()
        return (f"[{why}; the Python process was restarted, so its variables are gone. "
                "`session` and `od` are loaded again with the original model as v0]")

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        if self.conn:
            self.conn.close()


def _save_result(path: Path, r, route: str | None) -> None:
    np.savez(path, status=r.status, **{k: getattr(r, k) for k in ("obj", "x", "bound", "row_dual", "reduced_cost")
                                       if getattr(r, k) is not None})
    path.with_suffix(".route").write_text(route or "")


class _CallStopped(BaseException):  # raised in the worker's code at its call's limit (BaseException: code's own
    pass                            # `except Exception` must not swallow it)


def _cap(text: str) -> str:
    if len(text) <= OUT_CAP:
        return text
    cut = len(text) - OUT_CAP
    return (text[:OUT_CAP - 1500] + f"\n[... {cut} characters cut: print less, or summarise in code (a count, the "
            "top rows, a total) instead of printing whole results ...]\n" + text[-1500:])


def _confine(roots: list[str]) -> None:
    """Refuse file access outside ``roots`` and the Python installation, and refuse starting programs, for the rest of
    this process (an audit hook cannot be removed). Solver processes are started before, by the engine's own pool."""
    import sys
    import sysconfig
    import tempfile

    import optlens

    allowed = [os.path.realpath(p) for p in roots]
    allowed += [os.path.realpath(p) for p in {sys.prefix, sys.base_prefix, sys.exec_prefix, tempfile.gettempdir(),
                                              *sysconfig.get_paths().values(), *optlens.__path__} if p]
    allowed += ["/dev", "/proc", "/sys", "/etc"]  # devices, process info, locale and certificates (read by libraries)
    blocked = ("subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn", "os.exec", "os.startfile", "pty.spawn")

    def inside(path) -> bool:
        if isinstance(path, int):
            return True  # an already open file descriptor
        try:
            real = os.path.realpath(os.fsdecode(path))
        except (TypeError, ValueError):
            return True
        return any(real == a or real.startswith(a.rstrip(os.sep) + os.sep) for a in allowed)

    def hook(event, args):
        if event in blocked:
            raise PermissionError(f"{event}: starting programs is not allowed in this workspace")
        if event in ("open", "os.listdir", "os.scandir", "os.chdir", "shutil.copyfile", "os.rename", "os.remove") \
                and args and not inside(args[0]):
            raise PermissionError(f"{args[0]}: outside this workspace")

    sys.addaudithook(hook)


def _worker() -> None:
    import contextlib
    import ctypes
    import difflib
    import functools
    import io
    import time
    import traceback

    import optlens as od
    from optlens.session import (
        LARGE_MIP_IIS_BUDGET,
        MULTI_MODEL_TOOLS,
        TIME_LIMIT,
        TOOLS,
        Session,
        Version,
        chosen_solver,
    )

    conn = Client(os.environ.pop("CODE_WS_ADDR"), authkey=bytes.fromhex(os.environ.pop("CODE_WS_KEY")))
    confine = os.environ.pop("CODE_WS_CONFINE", None)
    prefer = os.environ.pop("CODE_WS_PREFER", None)
    only = bool(os.environ.pop("CODE_WS_ONLY_PREFER", "")) if prefer else chosen_solver() is not None
    # a solve that runs to its limit must still return inside the call's timeout, or the process restarts
    limit = float(os.environ.pop("CODE_WS_SOLVE_LIMIT", TIME_LIMIT))
    cap = os.environ.pop("CODE_WS_MAX_LIMIT", None)
    call = float(os.environ.pop("CODE_WS_TIMEOUT", TIMEOUT))
    iis_budget = max(min(LARGE_MIP_IIS_BUDGET, limit), limit - 5)
    if not cap:  # no client ceiling: a large MIP's IIS may use the call, whose deadline bounds it (E77: at 60 s the
        iis_budget = max(iis_budget, call - STEP_MARGIN - od.backends._GRACE - 2)  # 837k search did not finish)
    ns = {"session": Session({"v0": Version(od.load(os.environ["MODEL_FILE"]), None, "original model")},
                             os.environ.pop("CODE_WS_DOC") or None, prefer=prefer or chosen_solver(), only_prefer=only,
                             time_limit=limit, large_mip_iis_budget=iis_budget,
                             max_time_limit=float(cap) if cap else None),
          "od": od, "np": np, "MODEL_FILE": os.environ["MODEL_FILE"], "MODEL_DOC": os.environ["MODEL_DOC"],
          "TOOL_DOCS": {t["name"]: t["description"] for t in TOOLS + MULTI_MODEL_TOOLS}}
    if v0 := os.environ.pop("CODE_WS_V0", None):  # the caller's solve of the original model, and its solver
        with np.load(v0) as f:
            got = {k: (f[k].item() if f[k].ndim == 0 else f[k]) for k in f.files}
        ns["session"].versions["v0"].result = od.SolveResult(str(got.pop("status")), **got)
        ns["session"].route = os.environ.pop("CODE_WS_ROUTE", "") or None
    engine_calls: list[dict] = []

    def traced(name, method):
        @functools.wraps(method)  # inspect.signature and help() show the real method
        def call(*args, **kwargs):
            out = method(*args, **kwargs)
            engine_calls.append({"name": name, "input": {"args": list(args), **kwargs}, "output": str(out)[:20000]})
            return out
        return call

    for t in TOOLS:
        setattr(ns["session"], t["name"], traced(t["name"], getattr(ns["session"], t["name"])))
    if confine is not None:
        ns["session"].solved("v0")  # the engine's solver processes start here, before programs are refused
        _confine(confine.split(os.pathsep))
    base = set(ns) | {"__builtins__"}
    known: set[str] = set()  # versions whose solve the caller has
    saved_v0 = bool(os.environ.pop("CODE_WS_V0_SAVED", ""))  # the base solve is on disk for a restarted worker
    main = threading.get_ident()
    stop_lock, running = threading.Lock(), [False]

    def stop() -> None:  # at the call's limit: raise _CallStopped in the code, wherever it is
        with stop_lock:
            if running[0]:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(main), ctypes.py_object(_CallStopped))

    conn.send({"ready": True})
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        except _CallStopped:  # a stop that landed just as the last call ended
            continue
        buf, error = io.StringIO(), False
        session = ns["session"]
        if "versions" in msg:  # the caller's versions and solves (the MCP server's tools)
            session.restore_versions(msg["versions"])
            for vid, res in msg["results"].items():
                if vid in session.versions and session.versions[vid].result is None:
                    session.versions[vid].result = res
            known |= set(msg["results"])
        limit = msg.get("timeout")
        timer = None
        if limit:
            od.backends.set_step_deadline(time.time() + limit - STEP_MARGIN)
            timer = threading.Timer(max(1.0, limit - STOP_MARGIN), stop)
            timer.daemon = True
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                try:
                    with stop_lock:
                        running[0] = True
                    if timer:
                        timer.start()
                    exec(compile(msg["code"], "<run_python>", "exec"), ns)
                finally:
                    with stop_lock:
                        running[0] = False
                        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(main), None)  # one not yet raised
                    if timer:
                        timer.cancel()
            except od.StepTimeUsed as e:
                error = True
                od.backends.kill_busy()
                print(f"[this call's {limit:.0f} s ran out before the next solve or search could start ({e}): what "
                      "printed above finished; variables, versions and solves are kept. Run that step in its own call]")
            except _CallStopped:
                error = True
                running[0] = False
                od.backends.kill_busy()
                print(f"[stopped at {limit:.0f} s, this call's limit: what printed above finished; variables, versions "
                      "and solves are kept. Give each long solve or search its own call]")
            except NameError as e:
                error = True
                traceback.print_exc(limit=-1)
                user = [n for n in ns if n not in base]
                close = difflib.get_close_matches(getattr(e, "name", "") or "", [*user, *PRELOADED], n=1)
                if close:
                    print(f"[hint: `{close[0]}` exists in this process]")
            except BaseException:  # the agent's code may raise anything; it goes back as output
                error = True
                traceback.print_exc(limit=-3)
            if od.backends.CUTS and not error:
                asked, given = max(od.backends.CUTS, key=lambda c: c[0] - c[1])
                print(f"[this call's {limit:.0f} s were shared: {len(od.backends.CUTS)} solve(s) or searches got less "
                      f"time than asked (e.g. {asked:.0f} s asked, {given:.0f} s left); a result cut short says so. "
                      "Give each long step its own call]")
            od.backends.set_step_deadline(None)
        if not saved_v0 and (r0 := session.versions["v0"].result) is not None and r0.x is not None:
            try:  # a worker restarted after a time-out starts from it instead of solving the base again
                _save_result(Path(os.getcwd()) / "v0_result.npz", r0, session.route)
                saved_v0 = True
            except OSError:
                pass
        user = sorted(n for n in ns if n not in base and not n.startswith("_"))
        out = _cap(buf.getvalue()) + f"\n[in scope: {', '.join([*PRELOADED, *user])}]"
        made = {vid: v.result for vid, v in session.versions.items() if v.result is not None and vid not in known}
        known |= set(made)
        conn.send({"out": out, "error": error, "engine_calls": engine_calls[:],
                   **({"versions": session.export_versions(), "results": made} if "versions" in msg else {})})
        engine_calls.clear()


if __name__ == "__main__":
    _worker()
