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

TIMEOUT = 180.0
OUT_CAP = 10_000
PRELOADED = ("session", "od", "np", "MODEL_FILE", "MODEL_DOC", "TOOL_DOCS")


def tool_description(timeout: float = TIMEOUT, note: str = "") -> str:
    """The `run_python` tool's description: what is preloaded, and every engine method with its real signature. ``note``
    goes before the closing advice (the plugin says its versions are separate from its other tools')."""
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
        f"copy, `od.BACKENDS['{solver}'].solve(md)` returns `.status`, `.obj`, `.x`, and `md.row_names`, `md.col_names`, "
        "`md.row_lo`, `md.row_hi` describe it; `np`; `MODEL_FILE` and `MODEL_DOC`, the model and document paths. "
        + note + "Do several steps in one call and print only what you need. Methods of `session` (full description: "
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
                 confine: list[str] | None = None, prefer: str | None = None, only_prefer: bool = False):
        self.model_file = str(Path(model_path).resolve())
        self.doc_file = str(Path(doc_path).resolve()) if doc_path else ""
        self.timeout = timeout  # per call; None: TIMEOUT (read at call time)
        # confine: the code may read and write only inside the working directory, these extra paths (a pack, the
        # model and document) and Python's own installation, and may not start other programs (an evaluation must
        # not let an agent find tools or answers elsewhere on disk). None: no limits, as for a user's own machine.
        self.confine = None if confine is None else [str(Path(p).resolve()) for p in confine]
        # the worker's Session gets these; without them it takes the user's choice from OPTLENS_SOLVER, if any
        self.prefer, self.only_prefer = prefer, only_prefer
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.proc = self.conn = None
        self.last_engine_calls: list[dict] = []  # the session methods the last snippet called, for the attribution check
        self._start()

    def _start(self) -> None:
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
        env.update(MODEL_FILE=self.model_file, MODEL_DOC=self.doc_file, CODE_WS_ADDR=listener.address,
                   CODE_WS_KEY=key.hex(), CODE_WS_DOC=self.doc_file)
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
        if not self.conn.recv().get("ready"):
            raise RuntimeError(f"code worker failed to start; see {self.workdir / 'worker.log'}")

    def run_python(self, code: str) -> tuple[str, bool]:
        try:
            self.conn.send({"code": code})
            limit = self.timeout or TIMEOUT
            if not self.conn.poll(limit):
                return self._restart(f"timed out after {limit:.0f} s"), True
            r = self.conn.recv()
        except (EOFError, OSError):  # the worker died (out of memory on an 837k-row model): restart it, like a time-out
            return self._restart("the Python process stopped (it ran out of memory)"), True
        self.last_engine_calls = r["engine_calls"]
        return r["out"], r["error"]

    def _restart(self, why: str) -> str:
        self.close()
        self._start()
        self.last_engine_calls = []
        return (f"[{why}; the Python process was restarted, so its variables are gone. "
                "`session` and `od` are loaded again with the original model as v0]")

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        if self.conn:
            self.conn.close()


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
    import difflib
    import functools
    import io
    import traceback

    import numpy as np

    import optlens as od
    from optlens.session import MULTI_MODEL_TOOLS, TOOLS, Session, Version, chosen_solver

    conn = Client(os.environ.pop("CODE_WS_ADDR"), authkey=bytes.fromhex(os.environ.pop("CODE_WS_KEY")))
    confine = os.environ.pop("CODE_WS_CONFINE", None)
    prefer = os.environ.pop("CODE_WS_PREFER", None)
    only = bool(os.environ.pop("CODE_WS_ONLY_PREFER", "")) if prefer else chosen_solver() is not None
    ns = {"session": Session({"v0": Version(od.load(os.environ["MODEL_FILE"]), None, "original model")},
                             os.environ.pop("CODE_WS_DOC") or None, prefer=prefer or chosen_solver(), only_prefer=only),
          "od": od, "np": np, "MODEL_FILE": os.environ["MODEL_FILE"], "MODEL_DOC": os.environ["MODEL_DOC"],
          "TOOL_DOCS": {t["name"]: t["description"] for t in TOOLS + MULTI_MODEL_TOOLS}}
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
    conn.send({"ready": True})
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        buf, error = io.StringIO(), False
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                exec(compile(msg["code"], "<run_python>", "exec"), ns)
            except NameError as e:
                error = True
                traceback.print_exc(limit=-1)
                user = [n for n in ns if n not in base]
                close = difflib.get_close_matches(getattr(e, "name", "") or "", [*user, *PRELOADED], n=1)
                if close:
                    print(f"[hint: `{close[0]}` exists in this process]")
            except BaseException:  # noqa: BLE001 - the agent's code may raise anything; it goes back as output
                error = True
                traceback.print_exc(limit=-3)
        user = sorted(n for n in ns if n not in base and not n.startswith("_"))
        out = _cap(buf.getvalue()) + f"\n[in scope: {', '.join([*PRELOADED, *user])}]"
        conn.send({"out": out, "error": error, "engine_calls": engine_calls[:]})
        engine_calls.clear()


if __name__ == "__main__":
    _worker()
