"""Saved model context (optlens.context): key, validation, store, and the MCP tools that use them.
Run from the repo root: python -m unittest tests.test_context"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import optlens as od
from optlens.context import ContextStore, context_key, inventory, validate

HAS_MCP = importlib.util.find_spec("mcp") is not None
MODEL = Path(__file__).resolve().parent / "fixtures/ex_milp_tutorial__rhs_tighten__0.mps"
CONTEXT = {
    "overview": "Assign three workers to three jobs at the highest total match score.",
    "objective": "maximize the total match score",
    "indices": [{"name": "w", "description": "worker"}, {"name": "j", "description": "job"}],
    "families": [
        {"family": "job", "kind": "constraint", "meaning": "each job gets exactly one worker", "index_meaning": "job"},
        {"family": "assign", "kind": "variable", "meaning": "1 if the worker takes the job",
         "index_meaning": "worker, job"},
        {"family": "overtime", "kind": "variable", "meaning": "not in this model"},
    ],
}


class TestContext(unittest.TestCase):
    def setUp(self):
        self.md = od.load(MODEL)
        self.inv = inventory(self.md)

    def test_validate_reports_coverage_and_drops_unknown_families(self):
        clean, cov = validate(CONTEXT, self.inv)
        self.assertEqual(cov, {"undescribed": ["resource"], "unknown": ["overtime"]})
        self.assertEqual([f["family"] for f in clean["families"]], ["job", "assign"])
        self.assertEqual(clean["input_data"], [])  # optional fields default

    def test_validate_rejects_malformed(self):
        for bad in ([], {"overview": 3}, {"families": [{"family": "job", "kind": "row", "meaning": "x"}]},
                    {"indices": [{"name": "w"}]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate(bad, self.inv)

    def test_key_follows_structure_not_data(self):
        k = context_key(self.inv)
        edited = self.md.set_row_bounds(self.md.row_names[0], hi=123.0)  # a data change keeps the context
        self.assertEqual(context_key(inventory(edited)), k)

    def test_store_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(tmp)
            self.assertIsNone(store.load("m", "abc"))
            path = store.save("m", "abc", {"overview": "x"}, "doc.md")
            self.assertEqual(path.name, "m-abc.json")
            self.assertEqual(store.load("m", "abc"), {"overview": "x"})
            self.assertEqual(ContextStore(tmp).load("renamed", "abc"), {"overview": "x"})  # found by key


@unittest.skipUnless(HAS_MCP, "mcp not installed (optlens[mcp])")
class TestMcpContext(unittest.TestCase):
    def test_save_once_then_every_session_shows_it(self):
        from optlens.mcp_server import State

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"OPTLENS_SOLVER": "highs"}):
            doc = Path(tmp) / "tutorial.md"
            doc.write_text("Workers, jobs and the resource each job uses.")
            store = ContextStore(Path(tmp) / "ctx")
            first = State(store)
            out = first.open_model(str(MODEL), document=str(doc))
            self.assertIn("No saved model context", out)
            self.assertIn("constraint resource", out)
            saved, err = first.call("save_model_context", CONTEXT)
            self.assertFalse(err)
            self.assertIn("still undescribed: resource", saved)
            self.assertIn("not in this model, left out: overtime", saved)
            again = State(store).open_model(str(MODEL), document=str(doc))  # a later session
            self.assertIn("Saved model context", again)
            self.assertIn("each job gets exactly one worker", again)
            doc.write_text("A rewritten document.")  # the same model: its context stays, with a note
            changed = State(store).open_model(str(MODEL), document=str(doc))
            self.assertIn("each job gets exactly one worker", changed)
            self.assertIn("tutorial.md changed since this context was saved", changed)

    def test_save_before_open_is_an_error(self):
        from optlens.mcp_server import State

        out, err = State(ContextStore(tempfile.mkdtemp())).call("save_model_context", CONTEXT)
        self.assertTrue(err)


if __name__ == "__main__":
    unittest.main()
