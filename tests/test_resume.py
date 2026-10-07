"""A resumed conversation (E66): open_model restores the earlier session's versions, the method text tells the agent
to say so and to check version_history before saying nothing was done, version_history needs no argument, and "no
model is open" says how to get the versions back. Run from the repo root: python -m unittest tests.test_resume"""
import unittest
from pathlib import Path

import optlens as od
from optlens import prompts
from optlens.session import Session, Version

MODEL = Path(__file__).resolve().parent / "fixtures/ex_milp_tutorial__rhs_tighten__0.mps"


class TestResume(unittest.TestCase):
    def test_version_history_defaults_to_the_latest_version(self):
        s = Session({"v0": Version(od.load(MODEL), None, "original model")})
        self.assertTrue(s.version_history().startswith("v0: 0 step(s)"))
        row = s.get("v0").md.row_names[0]
        s.modify_and_resolve("v0", "looser", [{"action": "set_rhs", "name": row, "upper": 1e6}])
        self.assertTrue(s.version_history().startswith("v1: 1 step(s) and 1 change(s) from v0"))

    def test_the_method_says_to_report_restored_versions_and_check_before_denying(self):
        text = prompts.SERVER_INSTRUCTIONS
        self.assertIn("restored N versions", text)
        self.assertIn("tell the user, and check version_history before saying something\nwas not done earlier", text)
        self.assertIn('"no model is open" means the server restarted', text)

    def test_the_method_says_open_model_reads_gz_and_documents_are_read_not_shelled(self):
        text = prompts.SERVER_INSTRUCTIONS
        self.assertIn("open_model reads .lp,\n.mps and .gz files itself", text)
        self.assertIn("not by shelling out (cat, gzip -dc)", text)

    def test_no_model_open_says_the_server_may_have_restarted_and_how_to_restore(self):
        from optlens.mcp_server import State

        out, err = State().call("run_python", {"code": "print(1)"})
        self.assertTrue(err)
        self.assertIn("server may have restarted", out)
        self.assertIn("call open_model", out)
        self.assertIn("brings back the earlier session's versions", out)


if __name__ == "__main__":
    unittest.main()
