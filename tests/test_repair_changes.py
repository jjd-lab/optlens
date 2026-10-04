"""How modify_and_resolve reads a change's limits. Run from the repo root: python -m unittest tests.test_repair_changes"""
import unittest
from dataclasses import replace

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens.session import CHANGES_SHOWN, Session, Version


def flipped() -> od.ModelData:
    # floor: x <= 5 was meant to be x >= 5 (a flipped sense), which makes it clash with need: x >= 7
    return od.ModelData(
        name="m", minimize=True, obj=np.array([1.0]), obj_offset=0.0,
        A=sp.csr_matrix(np.array([[1.0], [1.0]])),
        row_lo=np.array([-od.INF, 7.0]), row_hi=np.array([5.0, od.INF]),
        col_lb=np.zeros(1), col_ub=np.full(1, 100.0), is_int=np.zeros(1, bool),
        col_names=("x",), row_names=("floor", "need"),
    )


class TestRepairChanges(unittest.TestCase):
    def setUp(self):
        self.s = Session({"v0": Version(flipped(), None, "original")})

    def test_null_removes_a_limit(self):
        out = self.s.modify_and_resolve("v0", "restore >=", [{"action": "set_rhs", "name": "floor", "lower": 5, "upper": None}])
        md = self.s.get("v1").md
        self.assertEqual((md.row_lo[0], md.row_hi[0]), (5.0, od.INF))
        self.assertIn("OPTIMAL", out)

    def test_left_out_side_is_unchanged(self):
        self.s.modify_and_resolve("v0", "lower only", [{"action": "set_rhs", "name": "floor", "lower": 5}])
        md = self.s.get("v1").md
        self.assertEqual((md.row_lo[0], md.row_hi[0]), (5.0, 5.0))

    def test_set_coef(self):
        # 2x <= 5 lets x reach 2.5 only, below need's 7; a zero coefficient drops x from floor
        self.s.modify_and_resolve("v0", "scale", [{"action": "set_coef", "name": "floor", "column": "x", "value": 2.0}])
        self.assertEqual(self.s.get("v1").md.A[0, 0], 2.0)
        out = self.s.modify_and_resolve("v0", "remove", [{"action": "set_coef", "name": "floor", "column": "x", "value": 0}])
        self.assertEqual(self.s.get("v2").md.A.nnz, 1)
        self.assertIn("OPTIMAL", out)

    def test_add_constraint(self):
        self.s.modify_and_resolve("v0", "drop", [{"action": "drop_constraint", "name": "floor"}])
        out = self.s.modify_and_resolve("v1", "cap", [{"action": "add_constraint", "name": "cap", "coefs": {"x": 1},
                                                       "lower": 8, "upper": None}])
        md = self.s.get("v2").md
        self.assertEqual((md.row_names[-1], md.row_lo[-1], md.row_hi[-1]), ("cap", 8.0, od.INF))
        self.assertIn("OPTIMAL", out)
        with self.assertRaises(ValueError):  # a name already in the model
            self.s.modify_and_resolve("v0", "dup", [{"action": "add_constraint", "name": "need", "coefs": {"x": 1}, "upper": 9}])

    def test_overview_lists_a_short_objective(self):
        self.assertIn("objective terms (1 nonzero): +1 x", self.s.get_model_overview())


class TestEditChecks(unittest.TestCase):
    def setUp(self):
        self.s = Session({"v0": Version(flipped(), None, "original")})

    def rejected(self, changes) -> str:
        out, err = self.s.call("modify_and_resolve", {"changes": changes})
        self.assertTrue(err, out)
        self.assertEqual(list(self.s.versions), ["v0"])  # nothing half-applied is kept
        return out

    def test_unknown_name_suggests_close_ones(self):
        out = self.rejected([{"action": "set_rhs", "name": "flor", "upper": 9}])
        self.assertIn("no constraint named 'flor'; did you mean floor?", out)

    def test_every_problem_is_reported_at_once(self):
        out = self.rejected([{"action": "set_bounds", "name": "y", "upper": 1},
                             {"action": "set_objective_coef", "name": "x", "value": "2"}])
        self.assertIn("change 1 (set_bounds y): no variable named 'y'", out)
        self.assertIn("change 2 (set_objective_coef x): value must be a number, got '2'", out)

    def test_crossed_limits(self):
        out = self.rejected([{"action": "set_rhs", "name": "need", "lower": 12, "upper": 10}])
        self.assertIn("lower 12 is above upper 10", out)

    def test_a_limit_can_move_in_two_steps(self):
        # need: x >= 7 moved to [12, 15] lower first: crossed in between, fine at the end
        self.s.modify_and_resolve("v0", "move", [{"action": "set_rhs", "name": "need", "upper": 15},
                                                 {"action": "set_rhs", "name": "need", "lower": 12}])
        md = self.s.get("v1").md
        self.assertEqual((md.row_lo[1], md.row_hi[1]), (12.0, 15.0))

    def test_a_failed_change_is_not_reported_again(self):
        out = self.rejected([{"action": "add_constraint", "name": "cap", "coefs": {"xx": 1}, "upper": 9},
                             {"action": "set_rhs", "name": "cap", "upper": 8}])
        self.assertIn("change 1 (add_constraint cap): no variable named 'xx'", out)
        self.assertNotIn("change 2", out)

    def test_try_options_checks_every_option_first(self):
        out, err = self.s.call("try_options", {"options": [
            {"label": "fine", "changes": [{"action": "set_rhs", "name": "floor", "lower": 5, "upper": None}]},
            {"label": "typo", "changes": [{"action": "set_rhs", "name": "flor", "upper": 9}]}]})
        self.assertTrue(err)
        self.assertIn("option 2 (typo), change 1 (set_rhs flor)", out)
        self.assertEqual(list(self.s.versions), ["v0"])

    def test_fractional_bound_on_an_integer_variable(self):
        s = Session({"v0": Version(replace(flipped(), is_int=np.ones(1, bool)), None, "original")})
        out, err = s.call("modify_and_resolve", {"changes": [{"action": "set_bounds", "name": "x", "upper": 7.5}]})
        self.assertTrue(err)
        self.assertIn("x is an integer variable; upper 7.5 is not a whole number", out)

    def test_a_later_change_sees_an_earlier_one(self):
        self.s.modify_and_resolve("v0", "add then edit", [
            {"action": "add_constraint", "name": "cap", "coefs": {"x": 1}, "upper": 9},
            {"action": "set_rhs", "name": "cap", "upper": 8}])
        self.assertEqual(self.s.get("v1").md.row_hi[-1], 8.0)

    def test_no_op_creates_no_version(self):
        out = self.s.modify_and_resolve("v0", "same", [{"action": "set_rhs", "name": "floor", "upper": 5}])
        self.assertIn("no change", out)
        self.assertEqual(list(self.s.versions), ["v0"])
        out = self.s.try_options("v0", [{"label": "same", "changes": [{"action": "set_coef", "name": "floor",
                                                                       "column": "x", "value": 1}]},
                                        {"label": "fix", "changes": [{"action": "set_rhs", "name": "floor",
                                                                      "lower": 5, "upper": None}]}])
        self.assertIn("same: no change from v0 (not solved)", out)
        self.assertIn("v1 fix: OPTIMAL", out)

    def test_limit_listing_is_capped(self):
        n = CHANGES_SHOWN + 5
        md = od.ModelData(
            name="m", minimize=False, obj=np.ones(n), obj_offset=0.0, A=sp.identity(n, format="csr"),
            row_lo=np.full(n, -od.INF), row_hi=np.ones(n), col_lb=np.zeros(n), col_ub=np.full(n, 10.0),
            is_int=np.zeros(n, bool), col_names=tuple(f"x{i}" for i in range(n)),
            row_names=tuple(f"cap{i}" for i in range(n)))
        out = Session({"v0": Version(md, None, "original")}).modify_and_resolve(
            "v0", "all up", [{"action": "set_rhs", "name": f"cap{i}", "upper": 2} for i in range(n)])
        self.assertEqual(out.count("limit cap"), CHANGES_SHOWN)
        self.assertIn("... (5 more limits changed)", out)


class TestLimitUse(unittest.TestCase):
    def test_slack_too_small_for_a_whole_step(self):
        # max 2a + 3b with 3a + 5b <= 7, integers: a = 2 spends 6 and leaves 1, too little for another a (3) or b (5);
        # a limit of 8 buys a + b = 5
        md = od.ModelData(
            name="m", minimize=False, obj=np.array([2.0, 3.0]), obj_offset=0.0, A=sp.csr_matrix(np.array([[3.0, 5.0]])),
            row_lo=np.array([-od.INF]), row_hi=np.array([7.0]), col_lb=np.zeros(2), col_ub=np.full(2, 5.0),
            is_int=np.ones(2, bool), col_names=("a", "b"), row_names=("budget",))
        out = Session({"v0": Version(md, None, "original")}).modify_and_resolve(
            "v0", "budget 8", [{"action": "set_rhs", "name": "budget", "upper": 8}])
        self.assertIn("limit budget: activity 6 of limit 7 (slack 1) -> 8 of limit 8 (slack 0)", out)
        self.assertIn("the smallest such step this limit blocked needs 3", out)

    def test_a_dropped_row_before_the_limit_shifts_its_index(self):
        # a drop and a limit change in one set; the limit's row index in the new model is one lower
        md = od.ModelData(
            name="m", minimize=False, obj=np.ones(2), obj_offset=0.0, A=sp.csr_matrix(np.array([[1.0, 0.0], [1.0, 1.0]])),
            row_lo=np.full(2, -od.INF), row_hi=np.array([1.0, 3.0]), col_lb=np.zeros(2), col_ub=np.full(2, 5.0),
            is_int=np.zeros(2, bool), col_names=("a", "b"), row_names=("fence", "budget"))
        out = Session({"v0": Version(md, None, "original")}).modify_and_resolve(
            "v0", "lift the fence, budget 4", [{"action": "drop_constraint", "name": "fence"},
                                               {"action": "set_rhs", "name": "budget", "upper": 4}])
        self.assertIn("limit budget: activity 3 of limit 3 (slack 0) -> 4 of limit 4 (slack 0)", out)


if __name__ == "__main__":
    unittest.main()
