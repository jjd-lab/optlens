"""Hotel pack hooks, for the planner chat agent built on optlens (pack.json, skill.md and hooks.py are its pack
format; the helpers in helpers.py work in any optlens session). startup: loads helpers.py into the workspace as
`hotel`, and in a two-stage session (rates.lp/.mps and params.json next to the booking model, as
`generate.py --pipeline` writes them) opens the rate plan as model `rates` and links it to the booking plan
(`ctx.link`); the agent then notices each solved rate-plan version and asks the planner before it is carried with
hotel.carry."""
from __future__ import annotations

from pathlib import Path

LOAD = r'''
import importlib.util as _iu, sys as _sys
_spec = _iu.spec_from_file_location("hotel", {helpers!r})
hotel = _iu.module_from_spec(_spec); _sys.modules["hotel"] = hotel; _spec.loader.exec_module(hotel)
hotel.MODEL_FILE = MODEL_FILE
print("hotel helpers loaded")
'''


def _stage_files(model_file: str) -> Path | None:
    folder = Path(model_file).parent
    rates = next((folder / n for n in ("rates.lp", "rates.lp.gz", "rates.mps", "rates.mps.gz") if (folder / n).exists()),
                 None)
    return rates if rates and (folder / "params.json").exists() else None


def startup(ctx):
    out, err = ctx.run_python(LOAD.format(helpers=str(ctx.path("helpers.py"))))
    if err:
        return f"[hotel pack: could not load its helpers: {out.strip()[-300:]}]"
    notes = ["Hotel helpers are loaded in run_python as `hotel` (see the hotel domain notes)."]
    rates = _stage_files(ctx.model_file)
    if rates is not None:
        out, err = ctx.run_python(f'print(session.add_model({str(rates)!r}, "rates", "stage 1: the rate plan whose '
                                  f'rates the booking plan v0 uses"))')
        if err:
            return f"[hotel pack: could not open the rate plan: {out.strip()[:300]}]"
        ctx.link("rates", "v0", "hotel.carry(session, RATES_VERSION, onto=BOOKING_PLAN)")
        notes.append("Two-stage session: v0 is the booking plan (stage 2); model `rates` is the rate plan (stage 1) "
                     "that set its rates.")
    return " ".join(notes)
