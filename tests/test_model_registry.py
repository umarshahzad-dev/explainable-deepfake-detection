import unittest
from pathlib import Path

from deepfake_detection.models.registry import (
    MODEL_REGISTRY,
    model_spec,
    resolve_model_run,
)
from deepfake_detection.common.paths import Workspace


class ModelRegistryTests(unittest.TestCase):
    def test_primary_and_archived_runs_resolve_to_existing_run_folders(self):
        workspace = Workspace()
        self.assertEqual(model_spec("xception-clean").category, "primary")
        for identifier, spec in MODEL_REGISTRY.items():
            if not spec.inference_enabled:
                continue
            with self.subTest(identifier=identifier):
                resolved = resolve_model_run(identifier, workspace)
                self.assertEqual(resolved, workspace.output(workspace.root / spec.artifact_path))
                self.assertTrue(resolved.is_dir())
                self.assertTrue((resolved / "config.json").is_file())

    def test_legacy_clean_run_name_remains_an_alias(self):
        workspace = Workspace()
        self.assertEqual(
            resolve_model_run("xcep-clean-s42", workspace),
            workspace.output(workspace.root / "models/primary/xception_clean"),
        )

    def test_focal_is_discoverable_but_refused_for_serving(self):
        spec = model_spec("xception-focal-legacy")
        self.assertEqual(spec.category, "legacy-unverified")
        self.assertFalse(spec.inference_enabled)
        self.assertTrue((Workspace().root / spec.artifact_path).is_file())
        with self.assertRaisesRegex(ValueError, "not enabled for inference"):
            resolve_model_run("xception-focal-legacy", Workspace())

    def test_custom_run_paths_still_work_with_workspace_containment(self):
        workspace = Workspace()
        self.assertEqual(resolve_model_run("local-experiment", workspace),
                         workspace.run("local-experiment"))
        self.assertEqual(resolve_model_run("archive/b0-ffpp-basic", workspace),
                         workspace.output(workspace.root / "models/archive/efficientnet_b0_ffpp_basic"))
        with self.assertRaises(ValueError):
            resolve_model_run("../outside", workspace)

    def test_inference_engine_refuses_focal_before_loading_weights(self):
        from deepfake_detection.inference.engine import InferenceEngine

        with self.assertRaisesRegex(ValueError, "not enabled for inference"):
            InferenceEngine.from_run("xception-focal-legacy", device="cpu")

    def test_api_health_resolves_primary_identifier(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from app.backend.api import create_app
        from tests.support import workspace as temporary_workspace

        workspace = Workspace(temporary_workspace(self))
        engine = type("Engine", (), {
            "metadata": {"run_id": "xcep-clean-s42", "weights_sha256": "fixture"}
        })()
        app = create_app(workspace=workspace, run="xception-clean")
        with patch("app.backend.api.InferenceEngine.from_run", return_value=engine):
            with TestClient(app) as client:
                response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["weights_sha256"], "fixture")

    def test_comparison_tool_uses_registered_model_identifiers(self):
        from tools.compare_models import RUNS

        self.assertTrue(RUNS)
        self.assertTrue(all(model_spec(identifier) for identifier, _ in RUNS))


if __name__ == "__main__":
    unittest.main()
