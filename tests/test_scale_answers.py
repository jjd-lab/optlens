"""Answers at scale (E66, 279k rows): sensitivity_report uses the version's own solve instead of solving it again, and
a result cut short by a limit says what to try next. Run from the repo root: python -m unittest tests.test_scale_answers"""
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens.session import TOOLS, Session, Version


def knapsack(is_int: bool = True) -> od.ModelData:
    """max 3a + 5b + 4c with at most two picked: b and c, 9."""
    return od.ModelData(
        name="k", minimize=False, obj=np.array([3.0, 5.0, 4.0]), obj_offset=0.0, A=sp.csr_matrix(np.ones((1, 3))),
        row_lo=np.array([-np.inf]), row_hi=np.array([2.0]), col_lb=np.zeros(3), col_ub=np.ones(3),
        is_int=np.full(3, is_int), col_names=("a", "b", "c"), row_names=("pick",))


def no_solve(*a, **k):
    raise AssertionError("solved the version again")


class TestSensitivityReusesTheSolve(unittest.TestCase):
    def test_a_solved_version_is_not_solved_again(self):
        for is_int in (True, False):
            with self.subTest(mip=is_int):
                s = Session({"v0": Version(knapsack(is_int), None, "original")})
                self.assertEqual(s.solved("v0").status, "OPTIMAL")
                with mock.patch.object(od.backends.Backend, "solve", no_solve):
                    out = s.sensitivity_report("constraints")
                self.assertIn("pick", out)
                self.assertNotIn("no sensitivity", out)


class TestLimitHints(unittest.TestCase):
    def test_marginal_value_says_what_to_try_when_a_re_solve_hits_its_limit(self):
        s = Session({"v0": Version(knapsack(), None, "original")})
        s.solved("v0")
        with mock.patch.object(od.backends.Backend, "solve", return_value=od.SolveResult("TIME_LIMIT")), \
                mock.patch.object(od, "gurobi_usable", return_value=False):
            out = s.marginal_value("pick", time_limit=20)
        self.assertIn("stopped at its 20 s limit; call again with a larger time_limit", out)
        self.assertNotIn("Gurobi", out)

    def test_sensitivity_says_what_to_try_and_names_gurobi_when_it_can_run(self):
        s = Session({"v0": Version(knapsack(), None, "original")})
        with mock.patch.object(od.backends.Backend, "solve", return_value=od.SolveResult("TIME_LIMIT")), \
                mock.patch.object(od, "gurobi_usable", return_value=True):
            out = s.sensitivity_report("constraints")
        self.assertIn("no sensitivity: status TIME_LIMIT", out)
        self.assertIn("larger time_limit", out)
        self.assertIn("solver gurobi", out)

    def test_the_mcp_server_offers_a_time_limit_on_both(self):
        from optlens.mcp_server import CALL_SOLVE_LIMIT, with_limits

        for name in ("marginal_value", "sensitivity_report"):
            tool = with_limits(next(t for t in TOOLS if t["name"] == name))
            self.assertIn(f"default {CALL_SOLVE_LIMIT:.0f}", tool["input_schema"]["properties"]["time_limit"]["description"])


if __name__ == "__main__":
    unittest.main()
