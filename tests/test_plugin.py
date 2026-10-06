"""The Claude Code plugin and the MCP server carry one method text, and the plugin installs by name.
Run from the repo root: python -m unittest tests.test_plugin"""
import json
import unittest
from pathlib import Path

from optlens import prompts

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugin"


class TestPlugin(unittest.TestCase):
    def test_the_skill_is_the_server_instructions(self):
        text = (PLUGIN / "skills/optlens/SKILL.md").read_text()
        front, body = text.split("---\n", 2)[1:]
        self.assertIn("name: optlens", front)
        self.assertEqual(body.strip(), prompts.SERVER_INSTRUCTIONS.strip())

    def test_the_instructions_name_the_tools_an_agent_starts_with(self):
        for tool in ("open_model", "save_model_context", "compute_iis", "fix_menu", "suspicious_values",
                     "modify_and_resolve", "compare_versions", "compare_models", "run_python"):
            self.assertIn(tool, prompts.SERVER_INSTRUCTIONS)

    def test_marketplace_and_manifest_agree(self):
        market = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
        manifest = json.loads((PLUGIN / ".claude-plugin/plugin.json").read_text())
        [entry] = market["plugins"]
        self.assertEqual(entry["name"], manifest["name"])
        self.assertTrue((ROOT / entry["source"]).joinpath(".claude-plugin/plugin.json").is_file())
        self.assertEqual(market["name"], "optlens")  # the install id is optlens@optlens

    def test_the_plugin_has_no_subagents(self):
        self.assertFalse((PLUGIN / "agents").exists())
        mcp = json.loads((PLUGIN / ".mcp.json").read_text())
        self.assertEqual(mcp["mcpServers"]["optlens"]["command"], "optlens-mcp")

    def test_the_plugin_passes_the_settings_through(self):
        env = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"]["optlens"]["env"]
        self.assertEqual(env["OPTLENS_CALL_LIMIT"], "${OPTLENS_CALL_LIMIT:-300}")
        # Claude Code stops a tool call at MCP_TOOL_TIMEOUT, which an environment may set to 60 s (a cloud session did,
        # and open_model timed out while its base solve ran); the server's own timeout overrides it
        timeout = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"]["optlens"]["timeout"]
        self.assertGreaterEqual(timeout, (300 + 10) * 1000)
        self.assertEqual(env["OPTLENS_SOLVER"], "${OPTLENS_SOLVER:-auto}")

    def test_the_instructions_say_how_to_load_deferred_tools(self):
        # Claude Code defers the tools; agents that searched for bare names never saw a description
        self.assertIn("mcp__plugin_optlens_optlens__open_model", prompts.SERVER_INSTRUCTIONS)
        self.assertIn("deferred", prompts.SERVER_INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
