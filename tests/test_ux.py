import io
import unittest
from contextlib import redirect_stdout
from tests.support import module,workspace


class UXTests(unittest.TestCase):
    def test_extract_progress_has_verification_counters(self):
        text=module(self,"presentation").extraction_detail(180,20000,47,2)
        self.assertIn("auto-verified 47",text)
        self.assertIn("review-needed 2",text)

    def test_long_command_summaries_have_required_fields(self):
        api=module(self,"presentation")
        cases={"extract":["Crops saved","Splits touched","Auto-verified","Review-needed","Disk used","Elapsed"],
               "train":["Best val AUC","Epochs run","Optimizer updates","Best checkpoint","Elapsed"],
               "evaluate":["Per-dataset AUC","Fake recall at 1% FPR","Fake recall at 5% FPR","Report path"],
               "calibrate":["Temperature","Threshold","Calibration FPR","Calibration recall"]}
        for command,fields in cases.items():
            text=api.format_summary(command,{})
            for field in fields: self.assertIn(field,text)

    def test_banner_prints_once_per_shell_and_quiet_suppresses(self):
        api=module(self,"presentation")
        ws=module(self,"paths").Workspace(workspace(self))
        output=io.StringIO()
        with redirect_stdout(output):
            api.first_banner("safe4gb.json","cpu",ws,shell_token="fixture",enabled=True)
            api.first_banner("safe4gb.json","cpu",ws,shell_token="fixture",enabled=True)
            with module(self,"progress").progress_options(quiet=True):
                api.first_banner("safe4gb.json","cpu",ws,shell_token="another",enabled=True)
        self.assertEqual(output.getvalue().count("experiment/ pipeline"),1)

    def test_progress_initial_line_is_immediate(self):
        output=io.StringIO()
        module(self,"progress").Progress("Video",5,stream=output,enabled=True)
        self.assertIn("Video 0/5",output.getvalue())
