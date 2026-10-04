"""Several models in one session (Session.add_model, compare_models). Run from the repo root: python -m unittest tests.test_multi_model"""
import importlib.util
import tempfile
import threading
import unittest
from pathlib import Path

import optlens as od
from optlens.session import Session, Version

MODEL = Path(__file__).resolve().parent / "fixtures/ex_milp_tutorial__rhs_tighten__0.mps"


class TestMultiModel(unittest.TestCase):
    def setUp(self):
        md = od.load(MODEL)  # infeasible: Monika's capacity 0.99
        self.tmp = tempfile.TemporaryDirectory()
        fixed = Path(self.tmp.name) / "fixed.mps"
        md.set_row_bounds("resource[Monika]", hi=1.0).write_mps(str(fixed))
        self.s = Session({"v0": Version(md, None, "as given")}, time_limit=30)
        self.out = self.s.add_model(str(fixed), "fixed")

    def tearDown(self):
        self.tmp.cleanup()

    def test_added_model_is_a_named_version_every_tool_takes(self):
        self.assertIn("same rows and columns as v0", self.out)
        self.assertEqual(self.s.solved("fixed").status, "OPTIMAL")
        self.assertIn("fixed", self.s.get_model_overview("fixed"))
        with self.assertRaises(ValueError):
            self.s.add_model(str(MODEL), "v1")  # version ids stay the session's own
        with self.assertRaises(ValueError):
            self.s.add_model(str(MODEL), "fixed")

    def test_compare_models_same_metrics_side_by_side(self):
        table = self.s.compare_models()
        self.assertIn("| status | INFEASIBLE | OPTIMAL |", table)
        self.assertIn("| job (rows at a limit) | - | 3 of 3 |", table)
        self.assertNotIn("(value; limit)", table)  # no one-row family in this model
        self.assertIn("compute_iis explains an infeasible one", table)

    def test_edits_after_an_added_model_get_fresh_ids_and_compare_defaults_to_the_latest(self):
        self.s.modify_and_resolve("fixed", "Monika at 2", [{"action": "set_rhs", "name": "resource[Monika]", "upper": 2}])
        self.assertIn("v2", self.s.versions)  # v0, fixed, then v2: ids never collide with names
        self.assertIn("fixed:", self.s.compare_versions("fixed")[:40])
        self.assertIn("v2", self.s.compare_versions("fixed")[:80])


@unittest.skipUnless(importlib.util.find_spec("mcp"), "mcp not installed (optlens[mcp])")
class TestParallelCalls(unittest.TestCase):
    def test_parallel_tool_calls_get_distinct_versions(self):  # analysts share one MCP server
        from optlens.mcp_server import State

        st = State()
        st.session = Session({"v0": Version(od.load(MODEL), None, "as given")}, time_limit=30)
        change = lambda hi: [{"action": "set_rhs", "name": "resource[Monika]", "upper": hi}]  # noqa: E731
        outs = []
        threads = [threading.Thread(target=lambda h=h: outs.append(st.call(
            "modify_and_resolve", {"base_version": "v0", "description": f"Monika {h}", "changes": change(h)})))
            for h in (1, 2, 3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(st.session.versions), ["v0", "v1", "v2", "v3"])
        self.assertFalse(any(err for _, err in outs))


if __name__ == "__main__":
    unittest.main()
