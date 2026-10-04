"""The hotel generator: the one-stage model is unchanged by the two-stage variant, and the rate plan feeds the booking
model. Run from packs/hotel/: python -m unittest discover -s tests -t ."""
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate as hotel  # noqa: E402

import optlens as od  # noqa: E402


class TestTwoStage(unittest.TestCase):
    def test_list_rate_levels_give_the_one_stage_model(self):
        p = hotel.Params(nights=10, linked=True)
        at_list = {f"h1_{r}_d{d}": 1 for r in hotel.ROOMS for d in range(10)}  # level 1 = 100 %
        self.assertEqual(hotel.build(hotel.with_levels(p, at_list))[0], hotel.build(p)[0])

    def test_pipeline_rate_plan_feeds_the_booking_model(self):
        folder = Path(tempfile.mkdtemp())
        out = hotel.pipeline(hotel.Params(nights=14), folder)
        self.assertEqual(out["rates"]["status"], "OPTIMAL")
        self.assertAlmostEqual(out["rates"]["objective"], 543081.51, places=2)
        levels = json.loads((folder / "levels.json").read_text())
        self.assertEqual(len(levels), 3 * 14)
        self.assertEqual(levels["h1_standard_d9"], 4)  # the event night: 150 % of list
        r = od.BACKENDS["highs"].solve(od.load(folder / "bookings.lp"), 120)
        self.assertEqual(r.status, "OPTIMAL")
        self.assertAlmostEqual(r.obj, 664372.34, places=2)

    def test_params_round_trip_through_json(self):
        p = hotel.Params(nights=7, hotels=2, linked=True)
        q = hotel.params_from_json(json.loads(json.dumps(asdict(p))))
        self.assertEqual(hotel.build(q)[0], hotel.build(p)[0])


if __name__ == "__main__":
    unittest.main()
