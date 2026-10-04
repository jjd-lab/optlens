"""A licensed solver (Gurobi) does every step itself; HiGHS and SCIP may stand in for each other and the result says so.
Run from the repo root: python -m unittest tests.test_licensed_solver"""
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens.session import Session, SolverUnavailable, Version

HAS_GUROBI = importlib.util.find_spec("gurobipy") is not None
LP_MODEL = Path(__file__).resolve().parent / "fixtures/ex_milp_tutorial__rhs_tighten__0.mps"


def bucket() -> od.ModelData:
    """A binary that the capacity row rounds to 0 (160 open <= 100) and a row that needs it at 1: gurobipy 13.0.3's
    IIS is min_open alone, which is feasible."""
    return od.ModelData(
        name="bucket", minimize=False, obj=np.array([1.0]), obj_offset=0.0,
        A=sp.csr_matrix(np.array([[160.0], [1.0]])), row_lo=np.array([-od.INF, 1.0]), row_hi=np.array([100.0, od.INF]),
        col_lb=np.zeros(1), col_ub=np.ones(1), is_int=np.ones(1, bool), col_names=("open",),
        row_names=("capacity", "min_open"),
    )


def no_open_source():
    """Patches that fail any HiGHS or SCIP call."""
    def refuse(*args, **kwargs):
        raise AssertionError("an open-source solver was called in a Gurobi session")
    return [mock.patch.object(od.BACKENDS[n], attr, side_effect=refuse)
            for n in ("highs", "scip") for attr in ("solve", "iis")] \
        + [mock.patch.object(od.BACKENDS["highs"], "farkas_rows", side_effect=refuse)]


@unittest.skipUnless(HAS_GUROBI, "gurobipy not installed")
class TestGurobiOnly(unittest.TestCase):
    def setUp(self):
        for p in no_open_source():
            p.start()
            self.addCleanup(p.stop)

    def test_a_feasible_native_iis_is_rejected(self):
        iis = od.get_iis(bucket(), od.BACKENDS["gurobi"])
        self.assertEqual(sorted(iis.rows), ["capacity", "min_open"])
        self.assertTrue(iis.method.startswith("deletion_filter:gurobi"))
        self.assertIn("gurobi's own IIS was rejected", iis.note)

    def test_every_step_runs_on_gurobi(self):
        for md in (bucket(), od.load(LP_MODEL)):
            s = Session({"v0": Version(md, None, "original model")}, prefer="gurobi", only_prefer=True)
            for tool in ("compute_iis", "feasibility_relaxation", "suspicious_values", "fix_menu"):
                with self.subTest(model=md.name, tool=tool):
                    out, err = s.call(tool, {})
                    self.assertFalse(err, out)
            out, err = s.call("sensitivity_report", {"kind": "constraints"})
            self.assertFalse(err, out)
            self.assertEqual(s.route, "gurobi")

    def test_the_bucket_conflict_names_both_rows(self):
        s = Session({"v0": Version(bucket(), None, "original model")}, prefer="gurobi", only_prefer=True)
        out, _ = s.call("compute_iis", {})
        self.assertIn("capacity", out)
        self.assertIn("min_open", out)

    def test_a_user_choice_that_cannot_run_stops_and_asks(self):
        s = Session({"v0": Version(bucket(), None, "original model")}, prefer="gurobi", only_prefer=True)
        with mock.patch.object(od.BACKENDS["gurobi"], "solve", side_effect=od.LicenseLimit("Model too large")):
            with self.assertRaises(SolverUnavailable) as e:
                s.solved("v0")
        self.assertIn("ask them before switching", str(e.exception))

    def test_another_solver_is_refused(self):
        s = Session({"v0": Version(bucket(), None, "original model")}, prefer="gurobi", only_prefer=True)
        out, err = s.call("compute_iis", {"solver": "highs"})
        self.assertTrue(err)
        self.assertIn("uses only gurobi", out)


class TestOpenSourceMix(unittest.TestCase):
    def test_a_step_on_the_other_open_source_solver_says_so(self):
        # A SCIP session: the MIP's LP relaxation is already infeasible, and HiGHS computes its IIS
        s = Session({"v0": Version(bucket(), None, "original model")}, prefer="scip")
        out, err = s.call("compute_iis", {})
        self.assertFalse(err, out)
        self.assertIn("computed by highs, not scip", out)

    def test_no_note_when_the_routed_solver_did_the_step(self):
        s = Session({"v0": Version(bucket(), None, "original model")}, prefer="highs")
        out, _ = s.call("compute_iis", {})
        self.assertNotIn("computed by", out)


if __name__ == "__main__":
    unittest.main()
