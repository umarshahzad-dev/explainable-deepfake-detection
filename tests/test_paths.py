import unittest
from pathlib import Path
from tests.support import module, workspace


class PathTests(unittest.TestCase):
    def test_even_explicit_output_cannot_modify_parent_project(self):
        api = module(self, "paths")
        root = workspace(self)
        with self.assertRaises(ValueError):
            api.Workspace(root).output(root.parent / "protected.json", explicit=True)

    def test_relative_config_file_ignores_process_working_directory(self):
        api = module(self, "config")
        root = workspace(self)
        (root / "configs").mkdir()
        (root / "configs/local.json").write_text('{"seed": 43}')
        cfg = api.load_config("configs/local.json", root=root)
        self.assertEqual(cfg["seed"], 43)

    def test_parent_write_is_rejected(self):
        api = module(self, "paths")
        root = workspace(self)
        paths = api.Workspace(root)
        with self.assertRaises(ValueError):
            paths.output(root.parent / "outside.json")

    def test_default_reports_use_runs_and_keep_lexical_path(self):
        api = module(self, "paths")
        root = workspace(self)
        paths = api.Workspace(root)
        self.assertEqual(paths.report("pilot"), root / "runs" / "reports" / "pilot")
        self.assertEqual(paths.output(root / "crops" / "x.png"), root / "crops" / "x.png")

    def test_traversal_run_name_is_rejected(self):
        api = module(self, "paths")
        paths = api.Workspace(workspace(self))
        with self.assertRaises(ValueError):
            paths.run("../parent")

    def test_configuration_resolves_relative_paths_against_experiment(self):
        api = module(self, "config")
        root = workspace(self)
        cfg = api.load_config(root=root, overrides={"raw_root": "../raw", "weights": "weights/model.safetensors"})
        self.assertEqual(Path(cfg["raw_root"]), (root / "../raw").absolute())
        self.assertEqual(Path(cfg["weights"]), root / "weights" / "model.safetensors")
        self.assertEqual(cfg["preprocessing"]["master_size"], 299)
        self.assertEqual(cfg["model"]["input_size"], 224)


if __name__ == "__main__":
    unittest.main()
