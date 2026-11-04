import io
import unittest
import os
import sys
from contextlib import redirect_stdout
from unittest.mock import patch
from tests.support import module, workspace


class ProgressTests(unittest.TestCase):
    def test_imported_unittest_does_not_silence_real_cli(self):
        api = module(self, "progress")
        presentation = module(self, "presentation")
        output = io.StringIO()
        ws = module(self, "paths").Workspace(workspace(self))
        with patch.dict(os.environ), patch.object(sys, "argv", ["tools/laptop_workflow.py", "extract"]), redirect_stdout(output):
            os.environ.pop("DEEPFAKE_EXPERIMENT_TEST_MODE", None)
            self.assertIn("unittest", sys.modules)
            api.Progress("Video", 1).finish(1)
            api.console("console fixture")
            presentation.print_summary("calibrate", {"temperature": 1, "threshold": .3})
            presentation.first_banner("fixture.json", "cpu", ws, shell_token="real-cli-fixture")
        self.assertIn("Video 0/1", output.getvalue())
        self.assertIn("console fixture", output.getvalue())
        self.assertIn("Summary: calibrate", output.getvalue())
        self.assertIn("experiment/ pipeline", output.getvalue())

    def test_explicit_test_marker_silences_even_cli_argv(self):
        output = io.StringIO()
        with patch.dict(os.environ, {"DEEPFAKE_EXPERIMENT_TEST_MODE": "1"}), patch.object(sys, "argv", ["tools/laptop_workflow.py"]), redirect_stdout(output):
            module(self, "progress").Progress("Video", 1).finish(1)
            module(self, "presentation").print_summary("calibrate", {})
        self.assertEqual(output.getvalue(), "")

    def test_unittest_launcher_detected_without_marker(self):
        api = module(self, "progress")
        with patch.dict(os.environ), patch.object(sys, "argv", [r"C:\Python\Lib\unittest\__main__.py"]):
            os.environ.pop("DEEPFAKE_EXPERIMENT_TEST_MODE", None)
            self.assertTrue(api.is_test_mode())
    def test_tty_uses_carriage_return_and_finishes_with_newline(self):
        api = module(self, "progress")
        class Terminal(io.StringIO):
            def isatty(self): return True
        output = Terminal()
        progress = api.Progress("Evaluated", 8, stream=output, enabled=True, interval=0, unit="faces")
        progress.update(5)
        progress.finish(8)
        self.assertIn("\rEvaluated 5/8 faces", output.getvalue())
        self.assertTrue(output.getvalue().endswith("\n"))

    def test_redirected_output_is_periodic_newlines(self):
        api = module(self, "progress")
        output = io.StringIO()
        progress = api.Progress("Video", 10, stream=output, enabled=True, interval=10, clock=lambda: 0)
        progress.update(1)
        progress.update(2)
        progress.finish(10)
        self.assertNotIn("\r", output.getvalue())
        self.assertEqual(len(output.getvalue().splitlines()), 2)

    def test_default_is_silent_in_unittest_and_quiet_disables_output(self):
        api = module(self, "progress")
        output = io.StringIO()
        api.Progress("Video", 1, stream=output).finish(1)
        with api.progress_options(quiet=True):
            api.Progress("Video", 1, stream=output, enabled=True).finish(1)
        self.assertEqual(output.getvalue(), "")

    def test_quiet_flag_works_before_or_after_subcommand(self):
        from tools.laptop_workflow import build_parser
        parser = build_parser()
        self.assertTrue(parser.parse_args(["--quiet", "extract", "--manifest", "x"]).quiet)
        self.assertTrue(parser.parse_args(["extract", "--manifest", "x", "--quiet"]).quiet)
