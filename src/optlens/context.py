"""Model context: the model's families joined to their business meaning, saved once and shown the same way every time.

The deterministic part is the inventory: every constraint and variable family with its count, senses or types and one
example name. The meanings (an "interpretation") come from an LLM outside this package (the host model
through the plugin's skill, or an agent's own call), so optlens needs no LLM SDK. What optlens adds: a key per document and
family structure, validation of an interpretation against the inventory, a store of saved interpretations, and one
renderer, so every agent and the plugin describe a model in the same words (standard meanings across sessions and
people)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .diagnose import base_name
from .model import ModelData

MAX_LINES = 40  # per section of the rendered context

RULES = """Index definitions below are ground truth: interpret variable and constraint indices from them, not from the LP data or expressions. A multi-index variable has one row per combination of all its indices; when a question is about one dimension, aggregate across the others rather than reporting each row as one entity. Families marked linking/definitional are structure (balances, links, definitions); they appear in conflicts because the chain passes through them, not because they are business levers."""


def inventory(md: ModelData) -> dict:
    def fams(names, counts_extra):
        out: dict[str, dict] = {}
        for i, n in enumerate(names):
            f = out.setdefault(base_name(n), {"count": 0, "example": n})
            f["count"] += 1
            counts_extra(f, i)
        return out

    def row_extra(f, i):
        sense = "=" if md.row_lo[i] == md.row_hi[i] else ">=" if np.isfinite(md.row_lo[i]) and not np.isfinite(md.row_hi[i]) \
            else "<=" if np.isfinite(md.row_hi[i]) and not np.isfinite(md.row_lo[i]) else "range"
        f.setdefault("senses", set()).add(sense)

    def col_extra(f, j):
        f.setdefault("types", set()).add("integer" if md.is_int[j] else "continuous")

    rows, cols = fams(md.row_names, row_extra), fams(md.col_names, col_extra)
    for f in rows.values():
        f["senses"] = sorted(f["senses"])
    for f in cols.values():
        f["types"] = sorted(f["types"])
    return {"constraints": rows, "variables": cols}


def _inventory_text(inv: dict) -> str:
    lines = []
    for kind, key in (("constraint", "constraints"), ("variable", "variables")):
        for name, f in sorted(inv[key].items()):
            extra = ", ".join(f.get("senses", f.get("types", [])))
            lines.append(f"- {kind} {name}: {f['count']} ({extra}), e.g. {f['example']}")
    return "\n".join(lines)


def render(ctx: dict) -> str:
    inv, interp, cov = ctx["inventory"], ctx["interpretation"], ctx["coverage"]
    meaning = {(f["kind"], f["family"]): f for f in (interp or {}).get("families", [])}
    lines = ["<ModelContext>", RULES, ""]
    if interp:
        lines += [f"Overview: {interp['overview']}", f"Objective: {interp['objective']}"]
        if interp.get("documented_result"):
            lines.append(f"Documented result (original model): {interp['documented_result']}")
        lines += ["", "Indices:"]
        lines += [f"- {i['name']}: {i['description']}" for i in interp["indices"]]
        if interp["input_data"]:
            lines += ["", "Input data:"] + [f"- {i['name']}: {i['description']}" for i in interp["input_data"]]
    if not ctx.get("names_meaningful", True):
        n_rows = sum(f["count"] for f in inv["constraints"].values())
        n_cols = sum(f["count"] for f in inv["variables"].values())
        lines += ["", f"Names carry no family structure ({n_rows} constraints, {n_cols} variables, e.g. "
                      f"{next(iter(inv['constraints']), '')}, {next(iter(inv['variables']), '')}). Rows grouped by shape "
                      "(sense, number of terms, coefficients), largest first:"]
        lines += [f"- {label}: {count} rows, e.g. {example}" for label, count, example in ctx["shapes"]]
        lines.append("</ModelContext>")
        return "\n".join(lines)
    lines += ["", "Families in the model file (count, example name, meaning):"]
    for kind, key in (("constraint", "constraints"), ("variable", "variables")):
        items = sorted(inv[key].items())
        if len(items) > MAX_LINES:
            lines.append(f"- ({len(items)} {kind} families; showing the {MAX_LINES} largest)")
            items = sorted(items, key=lambda kv: -kv[1]["count"])[:MAX_LINES]
        for name, f in items:
            m = meaning.get((kind, name))
            extra = ", ".join(f.get("senses", f.get("types", [])))
            desc = ""
            if m:
                desc = f" — {m['meaning']}" + (f" Indices: {m['index_meaning']}" if m["index_meaning"] else "") + \
                       (" [linking/definitional]" if m["linking"] else "")
            lines.append(f"- {kind} {name}: {f['count']} ({extra}), e.g. {f['example']}{desc}")
    if cov and (cov["undescribed"] or cov["unknown"]):
        lines += ["", f"Coverage: no meaning found for {cov['undescribed'] or 'none'}; "
                      f"described but not in the model file: {cov['unknown'] or 'none'}."]
    if interp and interp.get("not_in_model"):
        lines.append(f"Defined in the code but absent from this model file: {', '.join(interp['not_in_model'])}.")
    lines.append("</ModelContext>")
    return "\n".join(lines)


# ---- saved interpretations (the plugin's context cache) ----

SCHEMA = 1
FIELDS = {"overview": str, "objective": str, "documented_result": str, "indices": list, "input_data": list,
          "families": list, "not_in_model": list}


def context_key(inv: dict, doc: str) -> str:
    """Same document and family structure, same key: data edits and model versions reuse one saved context, and a
    family added later shows up as undescribed instead of breaking the match."""
    structure = sorted((k, n) for k in ("constraints", "variables") for n in inv[k] if n != "user_added_target")
    return hashlib.sha256((json.dumps(structure) + doc).encode()).hexdigest()[:16]


def validate(interp: dict, inv: dict) -> tuple[dict, dict]:
    """(the interpretation, cleaned, and its coverage). Raises ValueError on a malformed one; families the model does
    not have are kept out and reported, families left out are reported as undescribed."""
    if not isinstance(interp, dict):
        raise ValueError("the context must be an object")
    out: dict = {}
    for k, t in FIELDS.items():
        v = interp.get(k, [] if t is list else "")
        if not isinstance(v, t):
            raise ValueError(f"{k} must be a {t.__name__}")
        out[k] = v
    for k in ("indices", "input_data"):
        for i in out[k]:
            if not (isinstance(i, dict) and isinstance(i.get("name"), str) and isinstance(i.get("description"), str)):
                raise ValueError(f"each entry of {k} needs a name and a description")
    real = {("constraint", n) for n in inv["constraints"]} | {("variable", n) for n in inv["variables"]}
    families, unknown = [], []
    for f in out["families"]:
        if not isinstance(f, dict) or f.get("kind") not in ("constraint", "variable") or not isinstance(f.get("family"), str) \
                or not isinstance(f.get("meaning"), str):
            raise ValueError("each family needs family (an exact name from the list), kind (constraint or variable) "
                             "and meaning")
        f = {"family": f["family"], "kind": f["kind"], "meaning": f["meaning"],
             "index_meaning": str(f.get("index_meaning", "")), "linking": bool(f.get("linking", False))}
        (families if (f["kind"], f["family"]) in real else unknown).append(f)
    out["families"] = families
    described = {(f["kind"], f["family"]) for f in families}
    coverage = {"undescribed": sorted(n for k, n in real - described), "unknown": sorted(f["family"] for f in unknown)}
    return out, coverage


class ContextStore:
    """Saved interpretations as readable JSON files, one per (document, family structure): by default
    .optlens/context/ in the working directory, so a team can commit, review and correct them; OPTLENS_CONTEXT_DIR
    moves them."""

    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root or os.environ.get("OPTLENS_CONTEXT_DIR") or Path.cwd() / ".optlens" / "context")

    def path(self, model_name: str, key: str) -> Path:
        return self.root / f"{model_name}-{key}.json"

    def load(self, model_name: str, key: str) -> dict | None:
        hits = sorted(self.root.glob(f"*-{key}.json")) if self.root.is_dir() else []
        p = self.path(model_name, key)
        p = p if p.exists() else (hits[0] if hits else None)
        if p is None:
            return None
        data = json.loads(p.read_text())
        return data.get("interpretation") if data.get("schema") == SCHEMA else None

    def save(self, model_name: str, key: str, interp: dict, source: str = "") -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.path(model_name, key)
        p.write_text(json.dumps({"schema": SCHEMA, "model": model_name, "key": key, "source": source,
                                 "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "interpretation": interp}, indent=1))
        return p
