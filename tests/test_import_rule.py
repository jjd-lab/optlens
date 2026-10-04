"""optlens must stay an open package that installs on its own: it imports only the standard library, its declared
dependencies and its optional extras (no LLM SDK, nothing outside the package).
Run from the repo root: python -m unittest tests.test_import_rule"""
import ast
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src/optlens"
# pyproject.toml: dependencies, then the extras (anyio comes with mcp)
ALLOWED = {"optlens", "numpy", "scipy", "highspy", "pyscipopt", "gurobipy", "pulp", "pyomo", "mcp", "anyio"}


class TestImportRule(unittest.TestCase):
    def test_optlens_imports_only_the_standard_library_and_its_dependencies(self):
        if not SRC.is_dir():
            self.skipTest("needs the source tree (an installed copy has no src/)")
        bad = []
        for path in sorted(SRC.glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else []
                bad += [f"{path.name}: {n}" for n in names
                        if n.split(".")[0] not in ALLOWED | set(sys.stdlib_module_names)]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
