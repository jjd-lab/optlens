"""Hotel helpers for an agent's Python workspace: business views of a solved version and a data check against the
model's documented parameters. The hotel pack loads this module into the workspace as `hotel`, and any
optlens session can import it:

  hotel.occupancy(session, "v0")      rooms in house per night and room type, against the rooms for sale
  hotel.revenue(session, "v0")        revenue by room type, and the share sold at discounted tiers
  hotel.rate_plan(session, "rates")   the rate plan's base rate per room type and night (two-stage sessions)
  hotel.data_check(session, "v0")     every input that differs from the model rebuilt from its parameters
  hotel.carry(session, "v3", onto="v0")  two-stage: the booking plan at rate-plan version v3's rates, as a new version
  hotel.describe(name)                a constraint or variable name in business words
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate as G  # noqa: E402

import optlens as od  # noqa: E402

MODEL_FILE: str | None = None  # the session's model file (set by the pack when it loads this module)
_SELL = re.compile(r"sell\(h(\d+)_(\w+?)_(\d)n_d(\d+)_(\w+?)_t(\d)\)")
_LEVEL = re.compile(r"rate_level\(h(\d+)_(\w+?)_d(\d+)_L(\d)\)")
WORDS = {
    "demand_limit": "forecast demand for {r} rooms, {s}-night stays arriving night {d}, {w} bookings at tier t{t}",
    "room_capacity": "{r} rooms for sale on night {d} (after overbooking and the group block)",
    "discount_cap": "discounted bookings (tiers t0-t1) of {r} rooms arriving night {d}",
    "rate_fence": "last-minute discounted bookings of {r} rooms on night {d} (not allowed)",
    "closed_to_one_night": "one-night arrivals of {r} rooms on event night {d} (not allowed)",
    "housekeeping": "check-outs the housekeeping team can turn over on the morning of night {d}",
    "occupancy_floor": "the brand's minimum rooms in house on night {d}",
    "group_occupancy": "the group's minimum rooms in house on night {d}, all hotels together",
    "group_discount_budget": "the group's budget for revenue given up at discounted tiers",
    "one_rate": "one base rate for {r} rooms on night {d}",
    "rate_step_up": "how far the {r} rate may rise from night {d0} to night {d}",
    "rate_step_down": "how far the {r} rate may fall from night {d0} to night {d}",
    "event_premium": "the minimum {r} rate on event night {d}",
    "room_type_ladder": "the {r} rate's minimum gap above the room type below on night {d}",
    "occupancy_promise": "the rate plan's minimum expected rooms sold on night {d}",
    "sell": "bookings of {r} rooms, {s}-night stays arriving night {d}, {w} window, tier t{t}",
    "rate_level": "{r} rooms on night {d} at {pct} of the list rate",
}


def describe(name: str) -> str:
    """A row or column name in business words (hotel h: 'at hotel h' when there are several)."""
    fam, _, idx = name.partition("(")
    idx = idx.rstrip(")")
    f = {"r": "", "s": "", "d": "", "w": "", "t": "", "d0": "", "pct": ""}
    m = re.search(r"(?:^|_)d(\d+)", idx)
    if m:
        f["d"], f["d0"] = m.group(1), str(int(m.group(1)) - 1)
    for r in G.ROOMS:
        if f"_{r}_" in f"_{idx}_":
            f["r"] = r
    if (m := _SELL.match(name)):
        f.update(s=m.group(3), w=m.group(5), t=m.group(6))
    if (m := re.search(r"_(\d)n_d\d+_(\w+?)_t(\d)", idx)):
        f.update(s=m.group(1), w=m.group(2), t=m.group(3))
    if (m := _LEVEL.match(name)):
        f["pct"] = f"{G.Params().rate_levels[int(m.group(4))]:.0%}"
    text = WORDS.get(fam, fam).format(**f)
    h = re.match(r"h(\d+)_", idx)
    return f"{text}, hotel {h.group(1)}" if h and h.group(1) != "1" else text


def _solved(session, version):
    v, r = session.get(version), session.solved(version)
    if r.x is None:
        raise ValueError(f"{version} has no plan ({r.status})")
    return v.md, r


def occupancy(session, version: str = "v0", nights: list[int] | None = None) -> str:
    """Rooms in house per night and room type (all hotels), against the rooms for sale that night."""
    md, r = _solved(session, version)
    inhouse = defaultdict(float)
    for name, x in zip(md.col_names, r.x):
        m = _SELL.match(name)
        if m and x > 0.5:
            rt, s, d = m.group(2), int(m.group(3)), int(m.group(4))
            for k in range(d, d + s):
                inhouse[k, rt] += x
    cap = defaultdict(float)
    for name, hi in zip(md.row_names, md.row_hi):
        if name.startswith("room_capacity("):
            rt, d = re.match(r"room_capacity\(h\d+_(\w+)_d(\d+)\)", name).groups()
            cap[int(d), rt] += hi
    ds = sorted({d for d, _ in cap}) if nights is None else nights
    rows = ["night | " + " | ".join(G.ROOMS) + " | total | of rooms for sale"]
    for d in ds:
        tot, c = sum(inhouse[d, rt] for rt in G.ROOMS), sum(cap[d, rt] for rt in G.ROOMS)
        rows.append(f"{d} | " + " | ".join(f"{inhouse[d, rt]:.0f}/{cap[d, rt]:.0f}" for rt in G.ROOMS)
                    + f" | {tot:.0f} | " + (f"{tot / c:.0%}" if c else "-"))
    return f"{version}: rooms in house / rooms for sale\n" + "\n".join(rows)


def revenue(session, version: str = "v0") -> str:
    """Revenue by room type, and the share of room-nights sold at the discounted tiers (t0, t1)."""
    md, r = _solved(session, version)
    rev, nights, disc = defaultdict(float), defaultdict(float), defaultdict(float)
    for name, c, x in zip(md.col_names, md.obj, r.x):
        m = _SELL.match(name)
        if m and x > 0.5:
            rt, s, t = m.group(2), int(m.group(3)), int(m.group(6))
            rev[rt] += c * x
            nights[rt] += s * x
            if t <= 1:
                disc[rt] += s * x
    lines = [f"{version}: revenue {sum(rev.values()):,.2f}"]
    lines += [f"{rt}: {rev[rt]:,.2f} from {nights[rt]:,.0f} room-nights, {disc[rt] / nights[rt]:.0%} discounted"
              for rt in G.ROOMS if nights[rt]]
    return "\n".join(lines)


def rate_plan(session, version: str = "rates") -> str:
    """The rate plan's base rate per room type and night (% of the list rate and $), hotel 1 unless asked."""
    md, r = _solved(session, version)
    p = G.Params()
    pick = {}
    for name, x in zip(md.col_names, r.x):
        m = _LEVEL.match(name)
        if m and x > 0.5:
            pick[int(m.group(1)), m.group(2), int(m.group(3))] = int(m.group(4))
    ds = sorted({d for _, _, d in pick})
    rows = ["night | " + " | ".join(G.ROOMS)]
    for d in ds:
        cells = [f"{p.rate_levels[pick[1, rt, d]]:.0%} ${p.base_rate[rt] * p.rate_levels[pick[1, rt, d]]:,.0f}"
                 for rt in G.ROOMS if (1, rt, d) in pick]
        rows.append(f"{d}{' (event)' if G.is_event(p, d) else ''} | " + " | ".join(cells))
    return f"{version}: base rate per night (hotel 1)\n" + "\n".join(rows)


def rate_levels(session, version: str) -> dict:
    """A solved rate-plan version's chosen level per (hotel, room type, night), as generate.levels_from returns."""
    md, r = _solved(session, version)
    return G.levels_from(md.col_names, r.x)


def carry(session, rates_version: str, onto: str = "v0", model_file: str | None = None) -> str:
    """Two-stage: carry a solved rate-plan version into the booking plan. The booking plan's revenue per booking and
    demand limits are recomputed at that version's rates (generate.py, parameters from params.json next to the model
    file) and applied as changes to the booking-plan version ``onto`` (the planner's current plan, or v0 for the
    original), so changes kept in that plan survive. The result is a new version of the booking plan, solved; the
    pair is recorded in session.carried[rates_version]. Returns the result line."""
    folder = Path(model_file or MODEL_FILE).parent
    p = G.params_from_json(json.loads((folder / "params.json").read_text()))
    base = session.get(onto).md
    with tempfile.NamedTemporaryFile("w", suffix=".lp", delete=False) as f:
        f.write(G.build(G.with_levels(p, rate_levels(session, rates_version)))[0])
    ref = od.load(f.name)
    Path(f.name).unlink()
    cols = {n: i for i, n in enumerate(ref.col_names)}
    changes = [{"action": "set_objective_coef", "name": n, "value": float(ref.obj[cols[n]])}
               for i, n in enumerate(base.col_names) if n in cols and abs(base.obj[i] - ref.obj[cols[n]]) > 1e-9]
    rows = {n: i for i, n in enumerate(ref.row_names)}
    changes += [{"action": "set_rhs", "name": n, "upper": float(ref.row_hi[rows[n]])}
                for i, n in enumerate(base.row_names)
                if n.startswith("demand_limit(") and n in rows and abs(base.row_hi[i] - ref.row_hi[rows[n]]) > 1e-9]
    vid, text = session._modify(onto, f"booking plan at the rates of {rates_version} (carried onto {onto})", changes)
    session.carried = {**getattr(session, "carried", {}), rates_version: vid}
    r, b0 = session.solved(vid), session.solved(onto)
    line = f"{vid}: the booking plan {onto} at the rates of {rates_version} ({len(changes)} prices and demand limits changed): {r.status}"
    if r.obj is not None:
        line += f", revenue {r.obj:,.2f}"
        if b0.obj is not None:
            line += f" against {onto}'s {b0.obj:,.2f} (change {r.obj - b0.obj:+,.2f})"
    return line


def _params_for(md, model_file: str | None, is_rates: bool) -> tuple:
    """The model's parameters: params.json (and levels.json) next to the model file, else the defaults with the
    nights, hotels and linked rows read from the names."""
    folder = Path(model_file).parent if model_file else None
    if folder and (folder / "params.json").exists():
        p = G.params_from_json(json.loads((folder / "params.json").read_text()))
        if (folder / "levels.json").exists() and not is_rates:
            p = G.with_levels(p, json.loads((folder / "levels.json").read_text()))
        return p, "params.json next to the model"
    nights = 1 + max(int(m.group(1)) for n in md.row_names if (m := re.search(r"_d(\d+)\)$", n)))
    hotels = max(int(m.group(1)) for n in md.row_names if (m := re.search(r"\(h(\d+)_", n)))
    linked = any(n.startswith("group_occupancy(") for n in md.row_names)
    return G.Params(nights=nights, hotels=hotels, linked=linked), "the documented defaults"


def data_check(session, version: str = "v0", model_file: str | None = None, shown: int = 20) -> str:
    """Every limit, bound and rate that differs from the same model rebuilt from its documented parameters, in
    business words: the exact way to find a data-entry error (a typed target, a dropped or added rule). The
    parameters come from params.json next to the model file (MODEL_FILE, or model_file for an added model), else
    the documented defaults; a booking plan rebuilt at another rate plan differs in its rates, as it should."""
    md = session.get(version).md
    is_rates = any(n.startswith("rate_level(") for n in md.col_names)
    p, source = _params_for(md, model_file or MODEL_FILE, is_rates)
    text = (G.build_rates(p) if is_rates else G.build(p))[0]
    with tempfile.NamedTemporaryFile("w", suffix=".lp", delete=False) as f:
        f.write(text)
    ref = od.load(f.name)
    Path(f.name).unlink()
    diffs = []
    rows, ref_rows = {n: i for i, n in enumerate(md.row_names)}, {n: i for i, n in enumerate(ref.row_names)}
    for n, i in rows.items():
        j = ref_rows.get(n)
        if j is None:
            diffs.append(f"added rule {n}: {describe(n)}")
            continue
        for side, a, b in (("lower", md.row_lo[i], ref.row_lo[j]), ("upper", md.row_hi[i], ref.row_hi[j])):
            if not (a == b or abs(a - b) <= 1e-6 * max(1.0, abs(b))):
                diffs.append(f"{n}: {side} {a:g}, documented {b:g} ({describe(n)})")
    diffs += [f"missing rule {n}: {describe(n)}" for n in ref_rows if n not in rows]
    cols = {n: i for i, n in enumerate(ref.col_names)}
    for i, n in enumerate(md.col_names):
        j = cols.get(n)
        if j is not None and abs(md.obj[i] - ref.obj[j]) > 1e-6 * max(1.0, abs(ref.obj[j])):
            diffs.append(f"{n}: revenue per booking {md.obj[i]:g}, documented {ref.obj[j]:g} ({describe(n)})")
    head = f"{version} against the model rebuilt from {source}: "
    if not diffs:
        return head + "no differences"
    out = head + f"{len(diffs)} difference(s)"
    if len(diffs) > shown:  # many: by family first, so a whole family that moves (a forecast, the rates) stands out
        fams = defaultdict(int)
        for d in diffs:
            fams[re.sub(r"^(added rule |missing rule )?(\w+).*", r"\2", d)] += 1
        out += "\nby family: " + ", ".join(f"{f} {n}" for f, n in sorted(fams.items(), key=lambda kv: -kv[1]))
        out += "\nfirst ones:"
    return out + "\n" + "\n".join(diffs[:shown]) + (f"\n... ({len(diffs) - shown} more)" if len(diffs) > shown else "")
