"""Reader for LP files as Gurobi (and CPLEX) write them, with no solver library.

HiGHS rejects names with brackets (``x[0,1]``), which gurobipy writes for every indexed variable and
constraint, so without this reader a Gurobi-written ``.lp`` needs gurobipy to load. Semantics follow Gurobi's
own reader: columns in order of first appearance, unnamed rows ``R<i>``, the objective's ``Constant`` term is
the objective offset, a range is the ``Rg<row>`` column Gurobi writes, and a negative upper bound alone keeps
the default lower bound 0. Features ``ModelData`` cannot hold raise ``UnsupportedModel``.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .model import INF, ModelData, UnsupportedModel


class LPParseError(ValueError):
    """The file is not LP format this reader understands."""


_OBJ = {"max": True, "maximize": True, "maximise": True, "maximum": True,
        "min": False, "minimize": False, "minimise": False, "minimum": False}
_HEADERS = {
    **{k: "objective" for k in _OBJ},
    **{k: "constraints" for k in ("subject to", "such that", "st", "s.t.", "st.")},
    "bounds": "bounds", "bound": "bounds",
    **{k: "binaries" for k in ("binaries", "binary", "bin")},
    **{k: "generals" for k in ("generals", "general", "gen", "integers")},
    "lazy constraints": "constraints", "user cuts": "cuts",
    **{k: "unsupported" for k in ("semi-continuous", "semi-continuous variables", "semis", "semi", "sos", "sos1",
                                  "sos2", "general constraints", "general constraint", "gen cons", "pwlobj")},
    "end": "end",
}
_TOKEN = re.compile(r"""
    (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<arrow>->)
  | (?P<sense><=|=<|>=|=>|<|>|=)
  | (?P<op>[-+*/^:\[\]])
  | (?P<name>[^\s\d.+\-*/^:<>=\[\]\\](?:[^\s+\-*/^:<>=\[\]\\]|\[[^\]\s]*\])*)
  """, re.VERBOSE)
_SENSE = {"<=": "<", "=<": "<", "<": "<", ">=": ">", "=>": ">", ">": ">", "=": "="}
_INF_WORDS = {"inf", "infinity"}


def _tokens(text: str, where: str) -> list[tuple[str, str]]:
    out, i = [], 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        m = _TOKEN.match(text, i)
        if not m:
            raise LPParseError(f"{where}: cannot read {text[i:i + 30]!r}")
        out.append((m.lastgroup, m.group()))
        i = m.end()
    return out


def _header(line: str) -> tuple[str | None, str, str]:
    """(section, keyword, rest of the line) when the line opens a section, else (None, '', line)."""
    words = " ".join(line.split()).lower()
    if words in _HEADERS:
        return _HEADERS[words], words, ""
    if words.split(" ")[0] in _OBJ and "multi-objectives" in words:
        return "unsupported", words, ""
    if line[:1].isspace():  # writers indent continuation lines; a header with content follows it on the line
        return None, "", line
    for kw in sorted(_HEADERS, key=len, reverse=True):
        if words.startswith(kw + " ") and _HEADERS[kw] in ("objective", "constraints"):
            return _HEADERS[kw], kw, line.strip()[len(kw):]
    return None, "", line


class _Reader:
    def __init__(self, name: str):
        self.name = name
        self.cols: dict[str, int] = {}
        self.minimize = True
        self.obj: dict[int, float] = {}
        self.offset = 0.0
        self.quad: dict[tuple[int, int], float] = {}
        self.rows: list[tuple[str | None, dict[int, float], str, float]] = []
        self.lb: dict[int, float] = {}
        self.ub: dict[int, float] = {}
        self.binary: set[int] = set()
        self.integer: set[int] = set()

    def col(self, name: str) -> int:
        return self.cols.setdefault(name, len(self.cols))

    # ---- sections ----

    def objective(self, toks: list, where: str) -> None:
        i = 2 if len(toks) > 1 and toks[0][0] == "name" and toks[1] == ("op", ":") else 0
        lin, const, i = self._expr(toks, i, where, quad_ok=True)
        if i != len(toks):
            raise LPParseError(f"{where}: unexpected {toks[i][1]!r} in the objective")
        for j, v in lin.items():
            self.obj[j] = self.obj.get(j, 0.0) + v
        self.offset += const

    def constraints(self, toks: list, where: str) -> None:
        i = 0
        while i < len(toks):
            label = None
            if toks[i][0] == "name" and i + 1 < len(toks) and toks[i + 1] == ("op", ":"):
                label, i = toks[i][1], i + 2
            if any(t == ("arrow", "->") for t in toks[i:i + 6]):
                raise UnsupportedModel(f"{self.name}: indicator constraint {label or ''}".rstrip())
            lin, const, i = self._expr(toks, i, where, quad_ok=False)
            if i >= len(toks) or toks[i][0] != "sense":
                raise LPParseError(f"{where}: constraint {label or len(self.rows)} has no sense")
            sense = _SENSE[toks[i][1]]
            rhs, i = self._number(toks, i + 1, where)
            self.rows.append((label, lin, sense, rhs - const))

    def bounds(self, toks: list, where: str) -> None:
        if len(toks) == 2 and toks[0][0] == "name" and toks[1][1].lower() == "free":
            j = self.col(toks[0][1])
            self.lb[j], self.ub[j] = -INF, INF
            return
        parts, i = [], 0  # alternating operands and senses: v op n, n op v, n op v op n
        while i < len(toks):
            if toks[i][0] == "sense":
                parts.append(_SENSE[toks[i][1]])
                i += 1
            elif toks[i][0] == "name" and toks[i][1].lower() not in _INF_WORDS:
                parts.append(("var", toks[i][1]))
                i += 1
            else:
                v, i = self._number(toks, i, where)
                parts.append(("num", v))
        if len(parts) == 3 and parts[0][0] == "var":
            var, sense, value = parts[0][1], parts[1], parts[2][1]
            pairs = [(sense, value)]
        elif len(parts) == 3 and parts[2][0] == "var":
            var, sense, value = parts[2][1], {"<": ">", ">": "<", "=": "="}[parts[1]], parts[0][1]
            pairs = [(sense, value)]
        elif len(parts) == 5 and parts[2][0] == "var" and parts[1] == parts[3] != "=":
            var = parts[2][1]
            flip = {"<": ">", ">": "<"}[parts[1]]
            pairs = [(flip, parts[0][1]), (parts[3], parts[4][1])]
        else:
            raise LPParseError(f"{where}: cannot read bound {' '.join(t[1] for t in toks)!r}")
        if var == "Constant" and var not in self.cols:  # Gurobi's objective-offset placeholder
            return
        j = self.col(var)
        for sense, value in pairs:
            if sense in (">", "="):
                self.lb[j] = value
            if sense in ("<", "="):
                self.ub[j] = value

    def kinds(self, toks: list, where: str, kind: str) -> None:
        for t in toks:
            if t[0] != "name":
                raise LPParseError(f"{where}: expected variable names in {kind}, got {t[1]!r}")
            (self.binary if kind == "binaries" else self.integer).add(self.col(t[1]))

    # ---- pieces ----

    def _number(self, toks: list, i: int, where: str) -> tuple[float, int]:
        sign = 1.0
        while i < len(toks) and toks[i][1] in "+-" and toks[i][0] == "op":
            sign, i = (-sign if toks[i][1] == "-" else sign), i + 1
        if i < len(toks) and toks[i][0] == "num":
            return sign * float(toks[i][1]), i + 1
        if i < len(toks) and toks[i][0] == "name" and toks[i][1].lower() in _INF_WORDS:
            return sign * INF, i + 1
        raise LPParseError(f"{where}: expected a number, got {toks[i][1] if i < len(toks) else 'end'!r}")

    def _expr(self, toks: list, i: int, where: str, quad_ok: bool) -> tuple[dict[int, float], float, int]:
        """Linear terms and constant up to a sense, a label or the end; a [ ... ] block is the quadratic part."""
        lin: dict[int, float] = {}
        const = 0.0
        while i < len(toks) and toks[i][0] != "sense":
            if toks[i][0] == "name" and i + 1 < len(toks) and toks[i + 1] == ("op", ":"):
                break  # the next constraint's label
            sign = 1.0
            while toks[i][0] == "op" and toks[i][1] in "+-":
                sign, i = (-sign if toks[i][1] == "-" else sign), i + 1
            if toks[i] == ("op", "["):
                if not quad_ok:
                    raise UnsupportedModel(f"{self.name}: quadratic constraint")
                i = self._quad(toks, i + 1, sign, where)
                continue
            coef = 1.0
            if toks[i][0] == "num":
                coef, i = float(toks[i][1]), i + 1
            if i < len(toks) and toks[i][0] == "name" and not (i + 1 < len(toks) and toks[i + 1] == ("op", ":")):
                if quad_ok and toks[i][1] == "Constant":
                    const += sign * coef
                else:
                    j = self.col(toks[i][1])
                    lin[j] = lin.get(j, 0.0) + sign * coef
                i += 1
            elif toks[i - 1][0] == "num":
                const += sign * coef
            else:
                raise LPParseError(f"{where}: unexpected {toks[i][1]!r}")
        return lin, const, i

    def _quad(self, toks: list, i: int, outer: float, where: str) -> int:
        terms = []
        while toks[i] != ("op", "]"):
            sign = 1.0
            while toks[i][0] == "op" and toks[i][1] in "+-":
                sign, i = (-sign if toks[i][1] == "-" else sign), i + 1
            coef = 1.0
            if toks[i][0] == "num":
                coef, i = float(toks[i][1]), i + 1
            if toks[i][0] != "name":
                raise LPParseError(f"{where}: expected a variable in the quadratic term, got {toks[i][1]!r}")
            a, i = self.col(toks[i][1]), i + 1
            if toks[i] == ("op", "^"):
                if toks[i + 1] != ("num", "2"):
                    raise LPParseError(f"{where}: only squares are allowed, got ^{toks[i + 1][1]}")
                b, i = a, i + 2
            elif toks[i] == ("op", "*"):
                if toks[i + 1][0] != "name":
                    raise LPParseError(f"{where}: expected a variable after '*'")
                b, i = self.col(toks[i + 1][1]), i + 2
            else:
                raise LPParseError(f"{where}: expected '^ 2' or '* var' in the quadratic term")
            terms.append((a, b, sign * coef))
        i += 1
        scale = outer
        if i < len(toks) and toks[i] == ("op", "/"):
            scale /= float(toks[i + 1][1])
            i += 2
        for a, b, v in terms:  # objective has 0.5 x'Qx: a square's coefficient doubles, a product splits in two
            if a == b:
                self.quad[a, a] = self.quad.get((a, a), 0.0) + 2 * scale * v
            else:
                self.quad[a, b] = self.quad.get((a, b), 0.0) + scale * v
                self.quad[b, a] = self.quad.get((b, a), 0.0) + scale * v
        return i

    # ---- result ----

    def model(self) -> ModelData:
        n, m = len(self.cols), len(self.rows)
        lb, ub = np.zeros(n), np.full(n, INF)
        for j, v in self.lb.items():
            lb[j] = v
        for j, v in self.ub.items():
            ub[j] = v
        for j in self.binary:
            lb[j], ub[j] = max(lb[j], 0.0), min(ub[j], 1.0)
        is_int = np.zeros(n, bool)
        is_int[list(self.binary | self.integer)] = True
        obj = np.zeros(n)
        for j, v in self.obj.items():
            obj[j] = v
        r, c, d = [], [], []
        row_lo, row_hi = np.full(m, -INF), np.full(m, INF)
        for i, (_, lin, sense, rhs) in enumerate(self.rows):
            for j, v in lin.items():
                r.append(i)
                c.append(j)
                d.append(v)
            if sense in (">", "="):
                row_lo[i] = rhs
            if sense in ("<", "="):
                row_hi[i] = rhs
        Q = None
        if any(self.quad.values()):
            k = list(self.quad)
            Q = sp.csc_matrix(([self.quad[x] for x in k], ([x[0] for x in k], [x[1] for x in k])), shape=(n, n))
        big = lambda a: np.where(np.abs(a) >= 1e20, np.copysign(INF, a), a)  # noqa: E731  (solver infinities)
        return ModelData(
            name=self.name, minimize=self.minimize, obj=obj, obj_offset=self.offset,
            A=sp.csr_matrix((d, (r, c)), shape=(m, n)), row_lo=big(row_lo), row_hi=big(row_hi), col_lb=big(lb),
            col_ub=big(ub),
            is_int=is_int, col_names=tuple(self.cols),
            row_names=tuple(label or f"R{i}" for i, (label, *_) in enumerate(self.rows)), Q=Q,
        )


def read_lp(path: str | Path, name: str | None = None) -> ModelData:
    """Read an LP file into ``ModelData``."""
    path = Path(path)
    rd = _Reader(name or path.name.split(".")[0])
    section, chunks, seen = None, [], set()

    def flush():
        try:
            _flush()
        except IndexError:
            raise LPParseError(f"{path.name}:{chunks[0][0]}: a statement ends early") from None
        chunks.clear()

    def _flush():
        if not chunks:
            return
        where = f"{path.name}:{chunks[0][0]}"
        if section == "bounds":  # one bound per line
            for ln, text in chunks:
                toks = _tokens(text, f"{path.name}:{ln}")
                if toks:
                    rd.bounds(toks, f"{path.name}:{ln}")
        else:
            toks = _tokens(" ".join(t for _, t in chunks), where)
            if section == "objective":
                rd.objective(toks, where)
            elif section == "constraints":
                rd.constraints(toks, where)
            elif section in ("binaries", "generals"):
                rd.kinds(toks, where, section)
            elif section is None and toks:
                raise LPParseError(f"{where}: content before the objective section")

    with open(path, encoding="utf-8", errors="replace") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.split("\\", 1)[0].rstrip()
            if not line.strip():
                continue
            new, kw, rest = _header(line)
            if new is None:
                chunks.append((ln, line))
                continue
            flush()
            if new == "unsupported":
                raise UnsupportedModel(f"{path.name}: section {kw!r}")
            if new == "end":
                section = "end"
                break
            if new == "objective":
                if "objective" in seen:
                    raise UnsupportedModel(f"{path.name}: more than one objective")
                rd.minimize = not _OBJ[kw]
            seen.add(new)
            section = new
            if rest.strip():
                chunks.append((ln, rest))
    flush()
    if "objective" not in seen:
        raise LPParseError(f"{path.name}: no objective section")
    return rd.model()
