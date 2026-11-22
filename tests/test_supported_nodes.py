"""The README's node lists are generated from graphwalk.py's rule tables (tools/supported_nodes.py)."""

import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("supported_nodes_under_test", ROOT / "tools" / "supported_nodes.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

CORE = tool.load_core()


class ReadmeLists(unittest.TestCase):
    def test_readme_holds_the_generated_lists(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual(
            tool.readme_block(readme),
            tool.render(CORE),
            "README.md is out of date: run python3 tools/supported_nodes.py --write",
        )

    def test_every_rule_table_is_listed_or_excluded(self):
        self.assertEqual(tool.unlisted_tables(CORE), [])

    def test_check_passes_on_the_repository(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(tool.main(["--check"]), 0)

    def test_render_is_deterministic(self):
        self.assertEqual(tool.render(CORE), tool.render(tool.load_core()))


class Rendering(unittest.TestCase):
    def test_known_classes_of_each_table_appear(self):
        text = tool.render(CORE)
        for table in (
            CORE.SAMPLER_CLASSES, CORE.CUSTOM_SAMPLERS, CORE.SAVE_CLASSES, CORE.PREVIEW_CLASSES, CORE.TEXT_ENCODERS,
            CORE.COND_PASSTHROUGH, CORE.COND_COMBINE, CORE.COND_BLANK, CORE.GUIDERS, CORE.NOISE_SEEDS,
            CORE.SAMPLER_NODES, CORE.SCHEDULERS, CORE.SIGMAS_PASSTHROUGH, CORE.SWITCHES, CORE.TEXT_PASSTHROUGH,
            CORE.VALUE_PASSTHROUGH, CORE.SHOW_PASSTHROUGH, CORE.OUTPUT_WIDGETS, CORE.MODEL_LOADERS,
            CORE.LATENT_SIZE_CLASSES,
        ):
            for cls in table:
                self.assertIn(f"`{cls}`", text)

    def test_every_switch_kind_has_a_rule_text(self):
        for cls, rule in CORE.SWITCHES.items():
            self.assertIn(f"`{cls}`: {tool._switch_rule(rule)}", tool.render(CORE))
        self.assertEqual(
            {rule.kind for rule in CORE.SWITCHES.values()}, {"bool", "soft", "first", "index", "inversed"}
        )

    def test_a_table_the_lists_do_not_read_is_reported(self):
        namespace = type("Module", (), {})()  # module-like: vars() shows the new table next to the real ones
        namespace.__dict__.update(vars(CORE))
        namespace.NEW_TABLE = {"SomeNode"}
        self.assertEqual(tool.unlisted_tables(namespace), ["NEW_TABLE"])


class WriteAndCheck(unittest.TestCase):
    def run_tool(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            status = tool.main(list(args))
        return status, out.getvalue(), err.getvalue()

    def readme(self, text):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "README.md"
        path.write_text(text, encoding="utf-8")
        return path

    def test_write_replaces_only_the_marked_part_and_check_then_passes(self):
        path = self.readme(f"before\n\n{tool.BEGIN}\nold list\n{tool.END}\n\nafter\n")
        self.assertEqual(self.run_tool("--readme", str(path), "--check")[0], 1)
        self.assertEqual(self.run_tool("--readme", str(path), "--write")[0], 0)
        text = path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("before\n\n" + tool.BEGIN))
        self.assertTrue(text.endswith(tool.END + "\n\nafter\n"))
        self.assertEqual(tool.readme_block(text), tool.render(CORE))
        self.assertEqual(self.run_tool("--readme", str(path), "--check")[0], 0)

    def test_write_twice_changes_nothing(self):
        path = self.readme(f"{tool.BEGIN}\n{tool.END}\n")
        self.run_tool("--readme", str(path), "--write")
        first = path.read_text(encoding="utf-8")
        status, out, _ = self.run_tool("--readme", str(path), "--write")
        self.assertEqual((status, path.read_text(encoding="utf-8")), (0, first))
        self.assertIn("up to date", out)

    def test_readme_without_markers(self):
        path = self.readme("no markers\n")
        self.assertEqual(self.run_tool("--readme", str(path), "--write")[0], 2)
        self.assertEqual(self.run_tool("--readme", str(path), "--check")[0], 1)

    def test_print_mode_writes_the_lists_to_stdout(self):
        status, out, _ = self.run_tool()
        self.assertEqual((status, out.rstrip("\n")), (0, tool.render(CORE)))


if __name__ == "__main__":
    unittest.main()
