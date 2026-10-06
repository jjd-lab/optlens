"""The optlens MCP server, in-process. Run from the repo root: python -m unittest tests.test_mcp_server"""
import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

HAS_MCP = importlib.util.find_spec("mcp") is not None
MODEL = Path(__file__).resolve().parent / "fixtures/ex_milp_tutorial__rhs_tighten__0.mps"


@unittest.skipUnless(HAS_MCP, "mcp not installed (optlens[mcp])")
class TestMcpServer(unittest.TestCase):
    def run_client(self, steps):
        import anyio
        from mcp.client import Client

        from optlens.mcp_server import build_server

        async def go():
            async with Client(build_server()) as client:
                return [await step(client) for step in steps]
        with mock.patch.dict(os.environ, {"OPTLENS_SOLVER": "highs"}):
            return anyio.run(go)

    def test_lists_open_model_and_every_session_tool(self):
        from optlens.session import TOOLS

        [tools] = self.run_client([lambda c: c.list_tools()])
        self.assertEqual({t.name for t in tools.tools}, {"open_model", "save_model_context", "add_model", "compare_models", "run_python"} | {t["name"] for t in TOOLS})

    def test_arguments_off_the_schema_return_the_schema_and_run_nothing(self):
        # what an agent sent when Claude Code had deferred the tools and it never saw their schemas
        bad = {"label": "fix", "changes": [{"constraint": "c1", "rhs": 1000}]}
        option = {"options": [{"label": "fix", "changes": [{"name": "c1", "lower": 1000}]}]}
        opened, res, tried = self.run_client([lambda c: c.call_tool("open_model", {"path": str(MODEL)}),
                                              lambda c: c.call_tool("modify_and_resolve", bad),
                                              lambda c: c.call_tool("try_options", option)])
        text = res.content[0].text
        self.assertFalse(opened.is_error)
        self.assertTrue(res.is_error)
        self.assertIn("unknown argument 'label'", text)
        self.assertIn("'action' is a required property", text)
        self.assertIn('"enum": ["set_rhs"', text)
        self.assertIn("options.0.changes.0: 'action' is a required property", tried.content[0].text)

    def test_the_server_sends_the_method_as_its_instructions(self):
        from optlens import prompts
        from optlens.mcp_server import build_server

        self.assertEqual(build_server().create_initialization_options().instructions, prompts.SERVER_INSTRUCTIONS)

    def test_the_call_limit_sets_every_limit_together(self):
        from optlens.mcp_server import call_limits

        # (run_python call, default solve, default large-MIP IIS, the most a time_limit may ask for)
        self.assertEqual(call_limits(""), (50.0, 45.0, 40.0, 45.0))  # within the 60 s of Claude Desktop and others
        self.assertEqual(call_limits("300"), (290.0, 45.0, 40.0, 285.0))  # the defaults stay; time_limit may ask more
        self.assertEqual(call_limits("5"), (20.0, 15.0, 10.0, 15.0))  # at least 30 s
        for bad in ("five minutes", "nan"):
            self.assertEqual(call_limits(bad), (50.0, 45.0, 40.0, 45.0))
        with mock.patch.dict(os.environ, {"OPTLENS_CALL_LIMIT": "120"}):
            self.assertEqual(call_limits(), (110.0, 45.0, 40.0, 105.0))

    def test_the_agent_is_told_the_limits(self):
        [tools, opened] = self.run_client([lambda c: c.list_tools(),
                                           lambda c: c.call_tool("open_model", {"path": str(MODEL)})])
        schemas = {t.name: t.input_schema for t in tools.tools}
        self.assertIn("default 45, at most 45 here.", schemas["fix_menu"]["properties"]["time_limit"]["description"])
        self.assertIn("at most 45 here.", schemas["compute_iis"]["properties"]["time_limit"]["description"])
        # the what-if tool takes a time_limit through this server only (the bench's agents see the session's schema)
        self.assertIn("default 45, at most 45 here.",
                      schemas["modify_and_resolve"]["properties"]["time_limit"]["description"])
        from optlens.session import TOOLS
        self.assertNotIn("time_limit", next(t for t in TOOLS if t["name"] == "modify_and_resolve")["input_schema"]["properties"])
        self.assertIn("solves stop at 45 s unless a tool's time_limit asks for more, up to 45 s; each tool call answers "
                      "within 60 s (OPTLENS_CALL_LIMIT)", opened.content[0].text)

    def test_run_python_keeps_variables_and_has_the_engine(self):
        from optlens.mcp_server import State

        state = State()
        try:
            state.call("open_model", {"path": str(MODEL)})
            # the workspace starts from open_model's solve of v0: no second solve of the original model
            out, err = state.call("run_python", {"code": "print(session.get('v0').result is not None, session.route)"})
            self.assertIn(f"True {state.session.route}", out)
            out, err = state.call("run_python", {"code": "r = session.solved('v0').status\nprint(r)"})
            self.assertFalse(err, out)
            self.assertIn("INFEASIBLE", out)
            out, err = state.call("run_python", {"code": "print(r, len(session.versions))"})
            self.assertIn("INFEASIBLE 1", out)
        finally:
            state.close_workspace()

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy not installed")
    def test_gurobi_chosen_in_open_model_reaches_run_python(self):
        from optlens.mcp_server import State

        state = State()
        try:
            opened, err = state.call("open_model", {"path": str(MODEL), "solver": "gurobi"})
            self.assertFalse(err, opened)
            self.assertIn("uses only gurobi", opened)
            self.assertIn("od.BACKENDS['gurobi']", opened)
            out, err = state.call("run_python", {"code": "session.solved('v0'); print(session.route, session.strict())"})
            self.assertIn("gurobi True", out)
        finally:
            state.close_workspace()

    def test_open_diagnose_and_fix(self):
        opened, iis, fixed = self.run_client([
            lambda c: c.call_tool("open_model", {"path": str(MODEL)}),
            lambda c: c.call_tool("compute_iis", {}),
            lambda c: c.call_tool("modify_and_resolve", {"changes": [
                {"action": "set_rhs", "name": "resource[Monika]", "upper": 1}]}),
        ])
        self.assertIn("status: INFEASIBLE", opened.content[0].text)
        self.assertIn("preferred: highs", opened.content[0].text)
        self.assertIn("resource[Monika]", iis.content[0].text)
        self.assertIn("status OPTIMAL", fixed.content[0].text)
        self.assertFalse(fixed.is_error)

    def test_a_tool_before_open_model_is_an_error_result(self):
        [r] = self.run_client([lambda c: c.call_tool("compute_iis", {})])
        self.assertTrue(r.is_error)
        self.assertIn("call open_model", r.content[0].text)


class TestPreferredSolverFallback(unittest.TestCase):
    """A preferred solver that cannot run the model (e.g. a size-limited Gurobi license) falls back to routing."""

    def test_license_limit_falls_back_and_says_so(self):
        import optlens as od
        from optlens.session import Session, Version

        md = od.load(MODEL)
        s = Session({"v0": Version(md, None, "original model")}, prefer="gurobi")
        with mock.patch.object(od.BACKENDS["gurobi"], "solve", side_effect=od.LicenseLimit("Model too large for size-limited license")):
            status = s.solved("v0").status
        self.assertEqual((status, s.route), ("INFEASIBLE", "highs"))
        self.assertIn("preferred solver gurobi not used (LicenseLimit", s.route_note)

    def test_preferred_solver_is_used_when_it_runs(self):
        import optlens as od
        from optlens.session import Session, Version

        s = Session({"v0": Version(od.load(MODEL), None, "original model")}, prefer="scip")
        self.assertEqual(s.solved("v0").status, "INFEASIBLE")
        self.assertEqual(s.route, "scip")
        self.assertIn("scip (preferred)", s.route_note)


if __name__ == "__main__":
    unittest.main()
