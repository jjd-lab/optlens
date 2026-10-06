"""attainable_limit, drop_test and try_options on a small model. Run from the repo root: python -m unittest tests.test_agent_tools"""
import unittest

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens.session import Session, Version


def model() -> od.ModelData:
    # target: x + y >= 12 conflicts with budget: 2x + 3y <= 20 and cap: x <= 5 (at most 5 + 10/3 = 8.33 reachable)
    return od.ModelData(
        name="m", minimize=True, obj=np.array([1.0, 1.0]), obj_offset=0.0,
        A=sp.csr_matrix(np.array([[1.0, 1.0], [2.0, 3.0], [1.0, 0.0]])),
        row_lo=np.array([12.0, -od.INF, -od.INF]), row_hi=np.array([od.INF, 20.0, 5.0]),
        col_lb=np.zeros(2), col_ub=np.full(2, 100.0), is_int=np.zeros(2, bool),
        col_names=("x", "y"), row_names=("target", "budget", "cap"),
    )


class TestAgentTools(unittest.TestCase):
    def setUp(self):
        self.s = Session({"v0": Version(model(), None, "original")})

    def test_attainable_limit_both_sides(self):
        out = self.s.attainable_limit(["target", "budget"])
        self.assertIn("lower limit 12 -> at most 8.33333333333333", out)
        self.assertIn("upper limit 20 -> at least 31", out)  # x = 5, y = 7: 10 + 21

    def test_time_limits_stay_within_the_cap_and_a_lowered_one_is_said(self):
        self.assertEqual(self.s._limit(120, 60), 120)  # no cap: the bench and optchat sessions
        self.assertEqual(self.s._limit(None, 300), 300)
        s = Session({"v0": Version(model(), None, "original")}, time_limit=45, max_time_limit=100)  # the MCP server's
        self.assertEqual(s._limit(None, 300), 45)  # an engine default (an LP's IIS search) within the solve limit
        self.assertEqual(s._limit(90, 45), 90)  # a call may ask for more, up to the ceiling
        self.assertEqual(s.route_note, "")
        out, err = s.call("attainable_limit", {"constraints": ["target"], "time_limit": 120})
        self.assertFalse(err, out)
        self.assertIn("at most 8.33333333333333", out)
        self.assertIn("time_limit 120 s lowered to 100 s", out)
        out, err = s.call("modify_and_resolve", {"changes": [{"action": "set_rhs", "name": "budget", "upper": 40}],
                                                 "time_limit": 500})
        self.assertFalse(err, out)
        self.assertIn("status OPTIMAL", out)
        self.assertIn("time_limit 500 s lowered to 100 s", out)

    def test_drop_test(self):
        out = self.s.drop_test()
        self.assertIn("target (1 rows): dropping it restores feasibility", out)
        self.assertIn("budget (1 rows): dropping it restores feasibility", out)
        self.assertIn("cap (1 rows): still infeasible without it", out)  # y alone: 3y <= 20 cannot reach 12

    def test_marginal_value_prints_the_change_each_way(self):
        feasible = Session({"v0": Version(model().set_row_bounds("target", lo=6.0), None, "original")})
        out = feasible.marginal_value("target", delta=1)  # minimize x + y with x + y >= 6: the objective is the bound
        self.assertIn("bound 6 -> 7: OPTIMAL, objective 7 (change +1; 1 per unit the bound rises)", out)
        self.assertIn("bound 6 -> 5: OPTIMAL, objective 5 (change -1; 1 per unit the bound rises)", out)

    def test_why_not_counts_every_change_and_tallies_the_forced_index(self):
        # choose 2 of 4 by cost: y[a,1]=1, y[a,2]=2 are picked; forcing y[b,2] (4) drops y[a,2]
        md = od.ModelData(
            name="c", minimize=True, obj=np.array([1.0, 2.0, 3.0, 4.0]), obj_offset=0.0,
            A=sp.csr_matrix(np.ones((1, 4))), row_lo=np.array([2.0]), row_hi=np.array([od.INF]),
            col_lb=np.zeros(4), col_ub=np.ones(4), is_int=np.ones(4, bool),
            col_names=("y[a,1]", "y[a,2]", "y[b,1]", "y[b,2]"), row_names=("pick",))
        out = Session({"v0": Version(md, None, "original")}).why_not("y[b,2]", value=1)
        self.assertIn("variables changed: 2 of 4 (y: 1 up, 1 down)", out)
        self.assertIn("y with index 2 = 2: 0 up, 1 down", out)
        self.assertIn("y over its last index (1, 2), before -> after (O = 1, . = 0, - = none):\n  a: OO -> O.\n  b: .. -> .O", out)

    def test_compare_versions_by_family(self):
        md = od.ModelData(
            name="c", minimize=True, obj=np.array([1.0, 2.0, 3.0, 4.0]), obj_offset=0.0,
            A=sp.csr_matrix(np.ones((1, 4))), row_lo=np.array([2.0]), row_hi=np.array([od.INF]),
            col_lb=np.zeros(4), col_ub=np.ones(4), is_int=np.ones(4, bool),
            col_names=("y[a,1]", "y[a,2]", "y[b,1]", "y[b,2]"), row_names=("pick",))
        s = Session({"v0": Version(md, None, "original")})
        s.modify_and_resolve("v0", "force", [{"action": "set_bounds", "name": "y[b,2]", "lower": 1}])
        out = s.compare_versions()
        self.assertIn("v0: OPTIMAL, objective 3; v1: OPTIMAL, objective 5; change 2", out)
        self.assertIn("y: total 2 -> 2, nonzero 2 -> 2 of 4", out)
        self.assertIn("variables changed: 2 of 4 (y: 1 up, 1 down)", out)

    def test_compare_versions_after_a_dropped_constraint(self):
        # a waived rule (drop_constraint) leaves the versions with different rows
        s = Session({"v0": Version(model().set_row_bounds("target", lo=6.0), None, "original")})
        s.modify_and_resolve("v0", "no cap", [{"action": "drop_constraint", "name": "cap"},
                                              {"action": "set_rhs", "name": "target", "lower": 7}])
        out = s.compare_versions()
        self.assertIn("v0: OPTIMAL, objective 6; v1: OPTIMAL, objective 7; change 1", out)
        self.assertIn("rows only in the first version: 1; e.g. cap", out)

    def test_try_options(self):
        out = self.s.try_options(options=[{"label": "budget 31", "changes": [{"action": "set_rhs", "name": "budget", "upper": 31}]},
                                          {"label": "budget 30", "changes": [{"action": "set_rhs", "name": "budget", "upper": 30}]}])
        self.assertIn("budget 31: OPTIMAL", out)
        self.assertIn("budget 30: INFEASIBLE", out)

    def test_version_defaults_to_original(self):
        self.assertIn("status: INFEASIBLE", self.s.get_model_overview())
        self.assertIn("1 constraints match", self.s.query_constraints(family="budget"))


if __name__ == "__main__":
    unittest.main()
