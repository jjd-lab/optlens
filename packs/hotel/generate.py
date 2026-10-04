"""Hotel revenue management demo model (public: synthetic data, Apache-2.0).

A hotel group decides how many bookings to accept for every room type, arrival night, length of stay, booking
window and rate tier (the booking-limit model of hotel revenue management). Revenue is maximized subject to room
capacity (with an overbooking allowance and a group block), a cap on discounted tiers, a rate fence (no discounted
tiers last minute), event nights closed to one-night stays, a housekeeping limit on check-outs and an occupancy
floor. Document: model.md next to this file (the public hotel pack, packs/hotel/).

The model is a MILP (integer bookings) whose LP relaxation is nearly integral, so it solves in seconds at sizes
where an agent can no longer read it: about 190 rows and 180 columns per hotel-night, so 14 nights is a mid-size
demo and 365 nights x 12 hotels is about 830,000 rows. All data is synthetic and seeded.

Two-stage variant (a pipeline: one model's decisions are the next model's data). Stage 1, the rate plan,
picks each room type's base rate for each night from a ladder (85-150 % of the list rate), each level with its own
demand response, under business rules (room types keep their price order, rates move at most so much from one night
to the next, event nights carry a premium, an expected-occupancy promise). Stage 2 is the booking model above with
those rates and that demand. Without stage-1 levels the booking model is exactly the one-stage model.

Writes a CPLEX LP file with Pyomo-style names, `family(index_index)`, like the other demo models.

Usage:
  python packs/hotel/generate.py --nights 14 --hotels 1 --out hotel_14n.lp
  python packs/hotel/generate.py --nights 14 --pipeline OUTDIR
      (writes rates.lp, solves it through optlens, then levels.json and bookings.lp from its rate plan)
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ROOMS = ["standard", "deluxe", "suite"]
STAYS = [1, 2, 3]                                          # length of stay, nights
WINDOWS = ["early", "advance", "late", "lastminute"]       # 60+ days, 14-59, 3-13, 0-2 days before arrival
TIERS = [0, 1, 2, 3, 4]                                    # rate ladder, cheapest first


@dataclass
class Params:
    """Every number the model uses; the model document quotes these defaults."""
    nights: int = 14
    hotels: int = 1
    seed: int = 7
    room_count: dict = field(default_factory=lambda: {"standard": 120, "deluxe": 60, "suite": 20})
    overbooking: dict = field(default_factory=lambda: {"standard": 0.05, "deluxe": 0.03, "suite": 0.0})
    base_rate: dict = field(default_factory=lambda: {"standard": 140.0, "deluxe": 210.0, "suite": 420.0})
    tier_multiplier: list = field(default_factory=lambda: [0.80, 0.90, 1.00, 1.15, 1.30])
    tier_demand: list = field(default_factory=lambda: [1.45, 1.20, 1.00, 0.78, 0.60])   # demand at each tier
    base_arrivals: dict = field(default_factory=lambda: {"standard": 75.0, "deluxe": 32.0, "suite": 9.0})
    window_share: dict = field(default_factory=lambda: {"early": 0.20, "advance": 0.35, "late": 0.30,
                                                        "lastminute": 0.15})
    stay_share: dict = field(default_factory=lambda: {1: 0.50, 2: 0.30, 3: 0.20})
    stay_discount: dict = field(default_factory=lambda: {1: 0.0, 2: 0.05, 3: 0.10})       # per-night discount
    weekend_lift: float = 1.30            # Friday and Saturday arrivals
    event_lift: float = 1.60              # event nights
    event_every: int = 30                 # one event night every 30 nights, starting on night 9
    event_first: int = 9
    season_amplitude: float = 0.25        # yearly demand swing
    noise: float = 0.10                   # seeded day-to-day noise
    discount_share: float = 0.35          # discounted tiers' share of a room type's arrivals per night
    housekeeping_share: float = 0.55      # check-outs a day, as a share of all rooms
    occupancy_floor: float = 0.40         # share of rooms occupied every night
    group_block: dict = field(default_factory=lambda: {"standard": 10, "deluxe": 0, "suite": 0})
    # linked variant: group-wide rows across all hotels, so the model no longer splits into one piece per hotel
    linked: bool = False
    group_occupancy_floor: float = 0.60   # share of the group's rooms occupied every night, all hotels together
    discount_budget: float = 300.0        # group discount budget per hotel-night: revenue given up below the base tier
    # two-stage variant: stage 1 (the rate plan) picks a base-rate level per hotel, room type and night
    rate_levels: list = field(default_factory=lambda: [0.85, 1.00, 1.15, 1.30, 1.50])   # x the list base rate
    level_demand: list = field(default_factory=lambda: [1.20, 1.00, 0.84, 0.70, 0.56])  # arrivals at each level
    type_gap: dict = field(default_factory=lambda: {"deluxe": 40.0, "suite": 120.0})    # $ above the type below
    rate_step: float = 0.30               # largest night-to-night move, as a share of the list base rate
    event_premium: float = 1.30           # event nights: base rate at least this x the list rate
    occupancy_promise: float = 0.55       # expected rooms sold each night, as a share of all rooms
    avg_stay: float = 1.7                 # nights per arrival, for stage 1's expected rooms
    levels: dict | None = None            # stage 2: (hotel, room type, night) -> level index; None = list rates


def is_event(p: Params, d: int) -> bool:
    return d >= p.event_first and (d - p.event_first) % p.event_every == 0


def level(p: Params, h: int, r: str, d: int) -> int | None:
    """Stage 1's rate level for this hotel, room type and night (None: the list rates, one-stage model)."""
    return None if p.levels is None else p.levels[h, r, d]


def rate(p: Params, r: str, t: int, k: int | None = None) -> float:
    base = p.base_rate[r] if k is None else p.base_rate[r] * p.rate_levels[k]
    return round(base * p.tier_multiplier[t], 2)


def arrivals(p: Params) -> dict:
    """Expected arrivals at the list rate: (h, r, d) -> rooms (seeded; the same draws as the booking model's)."""
    rng = np.random.default_rng(p.seed)
    out = {}
    for h in range(p.hotels):
        size = 1.0 + 0.15 * h % 0.6                      # hotels differ a little in size of market
        for d in range(p.nights):
            lift = 1.0 + p.season_amplitude * math.sin(2 * math.pi * d / 365.0)
            if d % 7 in (4, 5):
                lift *= p.weekend_lift
            if is_event(p, d):
                lift *= p.event_lift
            for r in ROOMS:
                noise = 1.0 + rng.uniform(-p.noise, p.noise)
                out[h, r, d] = p.base_arrivals[r] * size * lift * noise
    return out


def demand_table(p: Params) -> dict:
    """Bookings available at each tier: (h, r, s, d, w, t) -> rooms; at stage 1's rate level when it is given."""
    A = arrivals(p)
    out = {}
    for h in range(p.hotels):
        for d in range(p.nights):
            for r in ROOMS:
                k = level(p, h, r, d)
                for s in STAYS:
                    for w in WINDOWS:
                        base = A[h, r, d] * p.stay_share[s] * p.window_share[w]
                        if k is not None:
                            base *= p.level_demand[k]
                        for t in TIERS:
                            out[h, r, s, d, w, t] = round(base * p.tier_demand[t], 2)
    return out


def hn(p: Params, h: int) -> str:
    return f"h{h + 1}_"


def build(p: Params) -> tuple[str, dict]:
    """The LP text and a size summary."""
    D = demand_table(p)
    obj, rows, ints = [], [], []
    total_rooms = sum(p.room_count.values())

    def sell(h, r, s, d, w, t):
        return f"sell({hn(p, h)}{r}_{s}n_d{d}_{w}_t{t})"

    def row(name, terms, sense, rhs):
        rows.append((name, terms, sense, rhs))

    def occupying(h, r, d):
        """Bookings in house on night d: arrivals on d-s+1 .. d that stay s nights."""
        return [sell(h, r, s, a, w, t) for s in STAYS for a in range(max(0, d - s + 1), d + 1)
                for w in WINDOWS for t in TIERS]

    for h in range(p.hotels):
        for d in range(p.nights):
            for r in ROOMS:
                for s in STAYS:
                    for w in WINDOWS:
                        for t in TIERS:
                            v = sell(h, r, s, d, w, t)
                            ints.append(v)
                            nightly = rate(p, r, t, level(p, h, r, d)) * (1 - p.stay_discount[s])
                            obj.append((round(nightly * s, 2), v))
                            row(f"demand_limit({hn(p, h)}{r}_{s}n_d{d}_{w}_t{t})", [(1, v)], "<=",
                                math.floor(D[h, r, s, d, w, t]))
                cap = math.floor(p.room_count[r] * (1 + p.overbooking[r])) - p.group_block[r]
                row(f"room_capacity({hn(p, h)}{r}_d{d})", [(1, v) for v in occupying(h, r, d)], "<=", cap)
                # discounted tiers (0 and 1) may fill at most this share of the room type's arrivals on night d
                disc = [sell(h, r, s, d, w, t) for s in STAYS for w in WINDOWS for t in TIERS if t <= 1]
                row(f"discount_cap({hn(p, h)}{r}_d{d})", [(1, v) for v in disc], "<=",
                    math.floor(p.discount_share * p.room_count[r]))
                # last-minute bookers never get the discounted tiers
                late = [sell(h, r, s, d, "lastminute", t) for s in STAYS for t in TIERS if t <= 1]
                row(f"rate_fence({hn(p, h)}{r}_d{d})", [(1, v) for v in late], "<=", 0)
                if is_event(p, d):
                    row(f"closed_to_one_night({hn(p, h)}{r}_d{d})",
                        [(1, sell(h, r, 1, d, w, t)) for w in WINDOWS for t in TIERS], "<=", 0)
            # check-outs on the morning of night d: stays that arrived d-s
            outs = [sell(h, r, s, d - s, w, t) for r in ROOMS for s in STAYS if d - s >= 0
                    for w in WINDOWS for t in TIERS]
            if outs:
                row(f"housekeeping({hn(p, h)}d{d})", [(1, v) for v in outs], "<=",
                    math.floor(p.housekeeping_share * total_rooms))
            occ_all = [v for r in ROOMS for v in occupying(h, r, d)]
            floor = math.ceil(p.occupancy_floor * total_rooms) - sum(p.group_block.values())
            row(f"occupancy_floor({hn(p, h)}d{d})", [(1, v) for v in occ_all], ">=", max(floor, 0))

    if p.linked:
        for d in range(p.nights):
            occ = [(1, v) for h in range(p.hotels) for r in ROOMS for v in occupying(h, r, d)]
            row(f"group_occupancy(d{d})", occ, ">=", math.ceil(p.group_occupancy_floor * total_rooms * p.hotels))
        # discount given = (base rate - tier rate) x (1 - stay discount) x nights, for tiers below the base (t0, t1)
        terms = [(round((rate(p, r, 2, level(p, h, r, d)) - rate(p, r, t, level(p, h, r, d))) * (1 - p.stay_discount[s])
                       * s, 2), sell(h, r, s, d, w, t))
                 for h in range(p.hotels) for d in range(p.nights) for r in ROOMS for s in STAYS for w in WINDOWS
                 for t in TIERS if t <= 1]
        row("group_discount_budget", terms, "<=", round(p.discount_budget * p.hotels * p.nights, 2))

    out = [f"\\* hotel_yield: {p.hotels} hotel(s), {p.nights} nights, seed {p.seed} *\\", "", "maximize", "revenue:"]
    out += [_term(c, v) for c, v in obj]
    out += ["", "subject to"]
    for name, terms, sense, rhs in rows:
        out.append(f"{name}:")
        out += [_term(c, v) for c, v in terms]
        out.append(f"{sense} {_num(rhs)}")
    out += ["", "general"] + ints + ["", "end", ""]
    nnz = sum(len(t) for _, t, _, _ in rows)
    return "\n".join(out), {"rows": len(rows), "cols": len(ints), "integers": len(ints), "nonzeros": nnz}


def build_rates(p: Params) -> tuple[str, dict]:
    """Stage 1, the rate plan: the LP text and a size summary. One binary per hotel, room type, night and rate level;
    the base rate is sum(level price x choice). Expected rooms at a level are the night's arrivals at that level's
    demand response times the average stay, capped by the rooms of that type."""
    A = arrivals(p)
    obj, rows, bins = [], [], []
    total_rooms = sum(p.room_count.values())
    K = range(len(p.rate_levels))

    def pick(h, r, d, k):
        return f"rate_level({hn(p, h)}{r}_d{d}_L{k})"

    def price(h, r, d, sign=1.0):
        return [(round(sign * p.base_rate[r] * p.rate_levels[k], 2), pick(h, r, d, k)) for k in K]

    def rooms(h, r, d, k):
        cap = math.floor(p.room_count[r] * (1 + p.overbooking[r])) - p.group_block[r]
        return round(min(cap, A[h, r, d] * p.level_demand[k] * p.avg_stay), 2)

    for h in range(p.hotels):
        for d in range(p.nights):
            for r in ROOMS:
                for k in K:
                    bins.append(pick(h, r, d, k))
                    obj.append((round(p.base_rate[r] * p.rate_levels[k] * rooms(h, r, d, k), 2), pick(h, r, d, k)))
                rows.append((f"one_rate({hn(p, h)}{r}_d{d})", [(1, pick(h, r, d, k)) for k in K], "=", 1))
                if d > 0:  # rates move at most rate_step x the list rate from one night to the next
                    step = round(p.rate_step * p.base_rate[r], 2)
                    rows.append((f"rate_step_up({hn(p, h)}{r}_d{d})", price(h, r, d) + price(h, r, d - 1, -1), "<=",
                                 step))
                    rows.append((f"rate_step_down({hn(p, h)}{r}_d{d})", price(h, r, d) + price(h, r, d - 1, -1),
                                 ">=", -step))
                if is_event(p, d):
                    rows.append((f"event_premium({hn(p, h)}{r}_d{d})", price(h, r, d), ">=",
                                 round(p.event_premium * p.base_rate[r], 2)))
            for lower, upper in (("standard", "deluxe"), ("deluxe", "suite")):  # room types keep their price order
                rows.append((f"room_type_ladder({hn(p, h)}{upper}_d{d})", price(h, upper, d) + price(h, lower, d, -1),
                             ">=", p.type_gap[upper]))
            rows.append((f"occupancy_promise({hn(p, h)}d{d})",
                         [(rooms(h, r, d, k), pick(h, r, d, k)) for r in ROOMS for k in K], ">=",
                         round(p.occupancy_promise * total_rooms, 2)))

    out = [f"\\* hotel_yield rate plan: {p.hotels} hotel(s), {p.nights} nights, seed {p.seed} *\\", "", "maximize",
           "expected_revenue:"]
    out += [_term(c, v) for c, v in obj]
    out += ["", "subject to"]
    for name, terms, sense, rhs in rows:
        out.append(f"{name}:")
        out += [_term(c, v) for c, v in terms]
        out.append(f"{sense} {_num(rhs)}")
    out += ["", "binary"] + bins + ["", "end", ""]
    nnz = sum(len(t) for _, t, _, _ in rows)
    return "\n".join(out), {"rows": len(rows), "cols": len(bins), "integers": len(bins), "nonzeros": nnz}


def levels_from(names, x) -> dict:
    """Stage 1's chosen level per (hotel, room type, night) from a solution: {"h1_standard_d5": k, ...}."""
    out = {}
    for n, v in zip(names, x):
        if n.startswith("rate_level(") and v > 0.5:
            key, k = n[len("rate_level("):-1].rsplit("_L", 1)
            out[key] = int(k)
    return out


def params_from_json(d: dict) -> Params:
    """Params from params.json (JSON keeps dictionary keys as strings: stay lengths are numbers again)."""
    d = {k: v for k, v in d.items() if k != "levels"}
    for k in ("stay_share", "stay_discount"):
        d[k] = {int(s): v for s, v in d[k].items()}
    return Params(**d)


def with_levels(p: Params, levels: dict) -> Params:
    """The booking model's parameters with stage 1's levels ({"h1_standard_d5": k} as levels_from returns)."""
    table = {}
    for key, k in levels.items():
        h, r, d = key.split("_")
        table[int(h[1:]) - 1, r, int(d[1:])] = int(k)
    return Params(**{**asdict(p), "levels": table})


def _num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else repr(round(float(x), 6))


def _term(c: float, v: str) -> str:
    return f"{'+' if c >= 0 else '-'}{_num(abs(c))} {v}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nights", type=int, default=14)
    ap.add_argument("--hotels", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, help="the model file (one-stage, or stage 2 with --levels)")
    ap.add_argument("--rates", type=Path, help="write stage 1, the rate plan, here")
    ap.add_argument("--levels", type=Path, help="stage 2: build the booking model at these rate levels (JSON)")
    ap.add_argument("--pipeline", type=Path, help="write rates.lp, solve it, then levels.json and bookings.lp here")
    ap.add_argument("--params", type=Path, help="write the parameters used as JSON here")
    ap.add_argument("--linked", action="store_true", help="add the group-wide rows (occupancy, discount budget)")
    a = ap.parse_args()
    p = Params(nights=a.nights, hotels=a.hotels, seed=a.seed, linked=a.linked)
    if a.params:
        a.params.write_text(json.dumps(asdict(p), indent=1))
    if a.pipeline:
        print(json.dumps(pipeline(p, a.pipeline)))
        return
    if a.rates:
        text, size = build_rates(p)
        a.rates.write_text(text)
        print(json.dumps({"rates": size}))
    if a.out:
        if a.levels:
            p = with_levels(p, json.loads(a.levels.read_text()))
        text, size = build(p)
        a.out.write_text(text)
        print(json.dumps(size))


def pipeline(p: Params, folder: Path) -> dict:
    """Both stages: the rate plan solved through optlens (HiGHS), its levels, and the booking model at them."""
    import optlens as od

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "params.json").write_text(json.dumps(asdict(p), indent=1) + "\n")  # to rebuild stage 2 from a new plan
    text, size1 = build_rates(p)
    (folder / "rates.lp").write_text(text)
    md = od.load(folder / "rates.lp")
    r = od.BACKENDS["highs"].solve(md, 600)
    if r.x is None:
        raise SystemExit(f"rate plan {r.status}")
    levels = levels_from(md.col_names, r.x)
    (folder / "levels.json").write_text(json.dumps(levels, indent=0, sort_keys=True) + "\n")
    text, size2 = build(with_levels(p, levels))
    (folder / "bookings.lp").write_text(text)
    return {"rates": {**size1, "status": r.status, "objective": r.obj}, "bookings": size2}


if __name__ == "__main__":
    main()
