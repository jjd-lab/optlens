"""The hotel helpers: the data check finds a typed input against the documented parameters; business views of a
plan. Run from packs/hotel/: python -m unittest discover -s tests -t ."""
import sys
import unittest
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))
import helpers as hotel  # noqa: E402

import optlens as od  # noqa: E402
from optlens.session import Session, Version  # noqa: E402

TWO = PACK / "examples/two_stage_14n"


class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.s = Session({"v0": Version(od.load(TWO / "bookings.lp.gz"), None, "booking plan")})
        self.s.add_model(str(TWO / "rates.lp.gz"), "rates")
        hotel.MODEL_FILE = str(TWO / "bookings.lp.gz")

    def test_data_check_finds_a_typed_target_and_nothing_else(self):
        self.assertTrue(hotel.data_check(self.s, "v0").endswith("no differences"))
        self.assertTrue(hotel.data_check(self.s, "rates").endswith("no differences"))
        vid, _ = self.s._modify("v0", "typo", [{"action": "set_rhs", "name": "housekeeping(h1_d6)", "upper": 11}])
        out = hotel.data_check(self.s, vid)
        self.assertIn("1 difference(s)", out)
        self.assertIn("housekeeping(h1_d6): upper 11, documented 110 (check-outs the housekeeping team can turn over "
                      "on the morning of night 6)", out)

    def test_data_check_without_params_uses_the_documented_defaults(self):
        hotel.MODEL_FILE = None  # the booking plan at stage-1 rates differs from the list-rate defaults in its rates
        out = hotel.data_check(self.s, "v0")
        self.assertIn("by family: sell ", out)  # every rate differs
        self.assertIn("demand_limit ", out)

    def test_business_views(self):
        self.assertIn("9 | 116/116 | 61/61 | 11/20 | 188 | 95%", hotel.occupancy(self.s, "v0", nights=[9]))
        self.assertIn("v0: revenue 664,372.34", hotel.revenue(self.s, "v0"))
        self.assertIn("9 (event) | 150% $210", hotel.rate_plan(self.s, "rates"))
        self.assertEqual(hotel.describe("occupancy_floor(h2_d3)"), "the brand's minimum rooms in house on night 3, hotel 2")


if __name__ == "__main__":
    unittest.main()
