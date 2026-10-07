"""An installed gurobipy that cannot run (no license, a version its license server rejects) must not block "auto": it is
left out of the preferred solvers, and a solve that fails that way falls back to HiGHS or SCIP. A user's own choice of
Gurobi still stops and asks (E66: such a gurobipy was preferred and every call failed). No Gurobi license is needed:
gurobipy is faked. Run from the repo root: python -m unittest tests.test_gurobi_unusable"""
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import optlens as od
from optlens import backends
from optlens.session import Session, SolverUnavailable, Version

MODEL = Path(__file__).resolve().parent / "fixtures/ex_milp_tutorial__rhs_tighten__0.mps"
NO_LICENSE, SIZE_LIMIT = 10009, 10010


def fake_gurobipy(env_error: str | None = None, optimize_errno: int | None = None) -> types.ModuleType:
    """A gurobipy whose Env fails with env_error, or whose optimize fails with optimize_errno."""
    gp = types.ModuleType("gurobipy")

    class GurobiError(Exception):
        def __init__(self, errno, message):
            super().__init__(message)
            self.errno = errno

    class GRB:
        OPTIMAL, INFEASIBLE, UNBOUNDED, INF_OR_UNBD, TIME_LIMIT = 2, 3, 5, 4, 9

        class Error:
            NO_LICENSE, SIZE_LIMIT_EXCEEDED = NO_LICENSE, SIZE_LIMIT

    class Model:
        Params = types.SimpleNamespace()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def optimize(self):
            raise GurobiError(optimize_errno, f"error {optimize_errno}")

    class Env(Model):
        def __init__(self, params=None):
            if env_error:
                raise GurobiError(NO_LICENSE, env_error)

        def dispose(self):
            pass

    gp.GurobiError, gp.GRB, gp.Env, gp.read = GurobiError, GRB, Env, lambda path, env=None: Model()
    return gp


class TestGurobiErrors(unittest.TestCase):
    def solve_with(self, gp):
        with mock.patch.dict(sys.modules, {"gurobipy": gp}):
            return od.BACKENDS["gurobi"]._solve(od.load(MODEL), 10)

    def test_gurobi_that_cannot_start_is_unusable(self):
        with self.assertRaises(od.SolverUnusable) as e:
            self.solve_with(fake_gurobipy(env_error="No compatible runtime available for version 13.0.3"))
        self.assertIn("No compatible runtime", str(e.exception))
        self.assertIsInstance(e.exception, od.LicenseLimit)  # so "auto" falls back the same way

    def test_a_license_error_in_the_solve_is_unusable_and_a_size_limit_stays_a_size_limit(self):
        with self.assertRaises(od.SolverUnusable):
            self.solve_with(fake_gurobipy(optimize_errno=NO_LICENSE))
        with self.assertRaises(od.LicenseLimit) as e:
            self.solve_with(fake_gurobipy(optimize_errno=SIZE_LIMIT))
        self.assertNotIsInstance(e.exception, od.SolverUnusable)


class TestGurobiProbe(unittest.TestCase):
    def setUp(self):
        self.saved = list(backends._GUROBI_PROBLEM)
        backends._GUROBI_PROBLEM.clear()

    def tearDown(self):
        backends._GUROBI_PROBLEM[:] = self.saved

    def test_installed_but_failing_gurobi_is_not_usable_and_the_probe_runs_once(self):
        probe = mock.Mock(return_value="GurobiError 10009: No Gurobi license found")
        with mock.patch("importlib.util.find_spec", return_value=object()), \
                mock.patch.object(backends, "_with_deadline", probe):
            self.assertFalse(od.gurobi_usable())
            self.assertIn("No Gurobi license", od.gurobi_problem())
        probe.assert_called_once()

    def test_a_license_server_that_does_not_answer_is_not_usable(self):
        with mock.patch("importlib.util.find_spec", return_value=object()), \
                mock.patch.object(backends, "_with_deadline", side_effect=backends._TimedOut):
            self.assertIn("no answer from Gurobi", od.gurobi_problem())

    def test_not_installed(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertEqual(od.gurobi_problem(), "gurobipy not installed")


class TestAutoFallsBack(unittest.TestCase):
    def test_auto_does_not_prefer_gurobi_that_cannot_run_and_open_model_says_why(self):
        from optlens import mcp_server

        problem = "GurobiError 10009: No compatible runtime available for version 13.0.3"
        with mock.patch.object(od, "gurobi_problem", return_value=problem), \
                mock.patch.object(od, "gurobi_usable", return_value=False), \
                mock.patch.object(od, "gurobipy_version", return_value="13.0.3"), \
                mock.patch.dict("os.environ", {"OPTLENS_SOLVER": "auto"}):
            self.assertNotIn("gurobi", mcp_server.available_solvers())
            self.assertEqual(mcp_server.preferred_solver(), (None, False))
            self.assertEqual(mcp_server.preferred_solver("gurobi"), ("gurobi", True))  # the user's choice stands
            line = mcp_server.solvers_line()
        self.assertIn("gurobipy 13.0.3 is installed but Gurobi cannot run", line)
        self.assertIn("No compatible runtime", line)
        self.assertIn('gurobipy==<major>.*', line)

    def test_usable_gurobi_is_preferred_and_shows_its_version(self):
        from optlens import mcp_server

        with mock.patch.object(od, "gurobi_problem", return_value=None), \
                mock.patch.object(od, "gurobi_usable", return_value=True), \
                mock.patch.object(od, "gurobipy_version", return_value="12.0.3"), \
                mock.patch.dict("os.environ", {"OPTLENS_SOLVER": "auto"}):
            self.assertEqual(mcp_server.preferred_solver(), ("gurobi", False))
            self.assertIn("gurobi (gurobipy 12.0.3)", mcp_server.solvers_line())

    def test_a_solve_that_finds_gurobi_unusable_falls_back_under_auto_and_stops_when_chosen(self):
        unusable = od.SolverUnusable("Gurobi cannot start: No Gurobi license found")
        s = Session({"v0": Version(od.load(MODEL), None, "original model")}, prefer="gurobi")
        with mock.patch.object(od.BACKENDS["gurobi"], "solve", side_effect=unusable):
            self.assertEqual(s.solved("v0").status, "INFEASIBLE")
        self.assertEqual(s.route, "highs")
        self.assertIn("preferred solver gurobi not used (SolverUnusable", s.route_note)

        s = Session({"v0": Version(od.load(MODEL), None, "original model")}, prefer="gurobi", only_prefer=True)
        with mock.patch.object(od.BACKENDS["gurobi"], "solve", side_effect=unusable), \
                self.assertRaises(SolverUnavailable) as e:
            s.solved("v0")
        self.assertIn("ask them before switching", str(e.exception))


if __name__ == "__main__":
    unittest.main()
