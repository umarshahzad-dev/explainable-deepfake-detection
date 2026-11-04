import io
import re
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch
from tests.support import module, workspace


class RunLogTests(unittest.TestCase):
    def test_captures_streams_partial_lines_and_appends(self):
        api = module(self, "runlog")
        ws = module(self, "paths").Workspace(workspace(self))
        path = ws.run("r") / "run.log"
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            for code in (0, 2):
                with api.RunLog(path, ["python", "-m", "tools.laptop_workflow", "train"], ws) as log:
                    print("stdout fixture")
                    import sys
                    sys.stderr.write("stderr fragment")
                    log.exit_status = code
        text = path.read_text(encoding="utf-8")
        self.assertEqual(text.count("START"), 2)
        self.assertIn("exit_status=2", text)
        self.assertIn("stderr fragment", text)
        self.assertIn("tools.laptop_workflow", text)
        self.assertTrue(all(re.match(r"\[\d{4}-\d{2}-\d{2}T", line) for line in text.splitlines()))

    def test_exception_records_failure_and_restores_streams(self):
        api = module(self, "runlog")
        ws = module(self, "paths").Workspace(workspace(self))
        import sys
        before = sys.stdout
        with self.assertRaises(ValueError):
            with api.RunLog(ws.run("r") / "run.log", ["fixture"], ws):
                raise ValueError("failure fixture")
        self.assertIs(sys.stdout, before)
        self.assertIn("exit_status=1", (ws.run("r") / "run.log").read_text())

    def test_cli_logs_success_and_failure_for_all_required_commands(self):
        from tools.laptop_workflow import main
        ws = module(self, "paths").Workspace(workspace(self))
        cases = [(["train", "--run", "r", "--manifests", "m"], ws.run("r")),
                 (["extract", "--manifest", "m"], ws.run("m")),
                 (["calibrate", "--run", "r", "--manifests", "m"], ws.run("r")),
                 (["evaluate", "--run", "r", "--manifests", "m"], ws.report("r-evaluation"))]
        for args, folder in cases:
            with self.subTest(command=args[0]), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), patch("tools.laptop_workflow.execute", return_value={}):
                self.assertEqual(main(args, workspace=ws), 0)
                self.assertIn("exit_status=0", (folder / "run.log").read_text())
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), patch("tools.laptop_workflow.execute", side_effect=ValueError("bad fixture")):
                self.assertEqual(main(args, workspace=ws), 2)
                self.assertIn("bad fixture", (folder / "run.log").read_text())
                self.assertIn("exit_status=2", (folder / "run.log").read_text())
