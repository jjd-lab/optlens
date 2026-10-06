"""The persistent Python workspace (the plugin's run_python and agents that write code).
Run from the repo root: python -m unittest tests.test_workspace"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from optlens import workspace
from optlens.workspace import CodeWorkspace

MODEL = Path(__file__).resolve().parent / "fixtures/milp_tutorial__0_RAP.mps"  # Gurobi modeling-examples, Apache-2.0


class TestCodeWorkspace(unittest.TestCase):
    def setUp(self):
        self.ws = CodeWorkspace(str(MODEL), Path(tempfile.mkdtemp()))

    def tearDown(self):
        self.ws.close()

    def run_ok(self, code: str) -> str:
        out, err = self.ws.run_python(code)
        self.assertFalse(err, out)
        return out

    def test_variables_and_versions_persist(self):
        self.run_ok("x = 5")
        self.assertIn("6", self.run_ok("print(x + 1)"))
        self.run_ok("r = session.try_options(options=[{'label': 'a', 'changes': [{'action': 'set_rhs', 'name': 'resource[Carlos]', 'upper': 2}]}])")
        self.assertIn("a: OPTIMAL", self.run_ok("print(r)"))

    def test_solves_fit_inside_the_call_timeout(self):
        self.assertEqual(self.run_ok("print(session.time_limit, session.max_time_limit)").split()[:2],
                         ["60.0", "None"])  # 180 s calls (bench, optchat): the default, no cap
        ws = CodeWorkspace(str(MODEL), Path(tempfile.mkdtemp()), timeout=25)
        try:
            out, err = ws.run_python("print(session.time_limit, session.large_mip_iis_budget)")
            self.assertFalse(err, out)
            self.assertEqual(out.split()[:2], ["15.0", "15.0"])
        finally:
            ws.close()

    def test_the_callers_base_solve_is_reused(self):
        # the MCP server solved the original model already: the worker starts from that result instead of solving it
        # again inside the call's timeout (a 90 s what-if plus a 45 s base solve overran a 100 s run_python call)
        import numpy as np

        from optlens import SolveResult

        fake = SolveResult("TIME_LIMIT", obj=123.0, x=np.zeros(3), bound=100.0)
        ws = CodeWorkspace(str(MODEL), Path(tempfile.mkdtemp()), base=("scip", fake))
        try:
            out, err = ws.run_python("r = session.solved('v0'); print(r.status, r.obj, r.bound, session.route)")
            self.assertFalse(err, out)
            self.assertEqual(out.split()[:4], ["TIME_LIMIT", "123.0", "100.0", "scip"])
        finally:
            ws.close()

    def test_a_longer_solve_limit_is_kept_inside_the_call_timeout(self):
        # the MCP server's OPTLENS_CALL_LIMIT=300: solves past the usual 60 s, still 10 s inside the call
        for timeout, solve_limit, expect in ((290, 285, ["280.0", "275.0"]), (50, 45, ["40.0", "40.0"])):
            ws = CodeWorkspace(str(MODEL), Path(tempfile.mkdtemp()), timeout=timeout, solve_limit=solve_limit)
            try:
                out, err = ws.run_python("print(session.time_limit, session.large_mip_iis_budget, "
                                         "session.max_time_limit)")
                self.assertFalse(err, out)
                self.assertEqual(out.split()[:3], expect + [f"{timeout - 15:.1f}"])  # a time_limit may ask up to this
            finally:
                ws.close()

    def test_engine_calls_are_recorded_for_the_attribution_check(self):
        self.run_ok("print(session.try_options(options=[{'label': 'a', 'changes': [{'action': 'set_rhs', 'name': 'resource[Carlos]', 'upper': 2}]}]))\nx = 1")
        calls = self.ws.last_engine_calls
        self.assertEqual([c["name"] for c in calls], ["try_options"])
        self.assertIn("a: OPTIMAL", calls[0]["output"])
        self.run_ok("print(x)")
        self.assertEqual(self.ws.last_engine_calls, [])

    def test_engine_methods_keep_their_signature(self):
        self.assertIn("(constraints=(), version='v0'", self.run_ok("import inspect\nprint(inspect.signature(session.attainable_limit))"))

    def test_numbers_through_od(self):
        out = self.run_ok("md = session.get('v0').md\nres = od.BACKENDS['highs'].solve(md)\nprint(res.status, round(res.obj))")
        self.assertIn("OPTIMAL 193", out)

    def test_the_attributes_the_description_names_exist(self):
        out = self.run_ok("v = session.get('v0'); md = v.md; r = session.solved('v0')\n"
                          "lp = md.lp_relaxation()\nprint(lp.is_int.any(), md.obj.shape == r.x.shape,"
                          " r.bound, r.gap, r.row_dual is None, r.reduced_cost is None, v.changes)")
        self.assertTrue(out.startswith("False True"), out)

    def test_name_error_names_the_closest_variable(self):
        self.run_ok("total = 1")
        out, err = self.ws.run_python("print(totl)")
        self.assertTrue(err)
        self.assertIn("[hint: `total` exists in this process]", out)
        self.assertIn("[in scope: session, od", out)

    def test_output_is_capped_with_a_note(self):
        out = self.run_ok("print('x' * 50000)")
        self.assertIn("characters cut: print less", out)
        self.assertLess(len(out), workspace.OUT_CAP + 500)

    def test_no_api_keys_inside(self):
        key = {"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "test-key")}
        with mock.patch.dict(os.environ, key):
            ws = CodeWorkspace(str(MODEL), Path(tempfile.mkdtemp()))
            try:
                out, _ = ws.run_python("import os\nprint(sorted(k for k in os.environ if 'KEY' in k))")
            finally:
                ws.close()
        self.assertIn("[]", out)

    def test_timeout_restarts_the_process(self):
        old, workspace.TIMEOUT = workspace.TIMEOUT, 2.0
        try:
            self.run_ok("y = 1")
            out, err = self.ws.run_python("import time\ntime.sleep(10)")
            self.assertTrue(err)
            self.assertIn("restarted", out)
            self.assertIn("True", self.run_ok("print('y' not in globals() and session is not None)"))
        finally:
            workspace.TIMEOUT = old


class TestConfinedWorkspace(unittest.TestCase):
    """A confined workspace (confine=[]): the agent's code works inside its folder, but cannot read elsewhere or run
    programs."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.ws = CodeWorkspace(str(MODEL), self.dir, confine=[])

    def tearDown(self):
        self.ws.close()

    def test_solving_and_own_files_work(self):
        out, err = self.ws.run_python("r = session.try_options(options=[{'label': 'a', 'changes': [{'action': 'set_rhs', "
                                      "'name': 'resource[Carlos]', 'upper': 2}]}])\nopen('note.txt', 'w').write('x')\n"
                                      "print(r, open('note.txt').read())")
        self.assertFalse(err, out)
        self.assertIn("a: OPTIMAL", out)

    def test_files_elsewhere_and_programs_are_refused(self):
        here = Path(__file__).resolve()
        # the temp directory is always allowed (solvers write there), so a checkout under it is not "elsewhere"
        if not str(here).startswith(os.path.realpath(tempfile.gettempdir()) + os.sep):
            out, err = self.ws.run_python(f"print(open({str(here)!r}).read()[:20])")
            self.assertTrue(err)
            self.assertIn("outside this workspace", out)
        out, err = self.ws.run_python("import os\nprint(os.listdir('/home'))")
        self.assertTrue(err)
        out, err = self.ws.run_python("import subprocess\nprint(subprocess.run(['ls', '/'], capture_output=True).stdout)")
        self.assertTrue(err)
        self.assertIn("starting programs is not allowed", out)
        out, err = self.ws.run_python("import os\nos.system('ls /')")
        self.assertTrue(err)


if __name__ == "__main__":
    unittest.main()
