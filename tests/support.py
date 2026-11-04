import importlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
MODULES = {'data': 'deepfake_detection.data.datasets', 'detector': 'deepfake_detection.data.face_detector', 'extraction': 'deepfake_detection.data.extraction', 'preprocessing': 'deepfake_detection.data.preprocessing', 'manifests': 'deepfake_detection.data.manifests', 'inventory': 'deepfake_detection.data.inventory', 'identity': 'deepfake_detection.data.identity', 'integrity': 'deepfake_detection.data.integrity', 'review': 'deepfake_detection.data.review', 'architecture': 'deepfake_detection.models.architecture', 'attention': 'deepfake_detection.models.attention', 'frequency': 'deepfake_detection.models.frequency', 'losses': 'deepfake_detection.models.losses', 'model': 'deepfake_detection.models.model_loader', 'pretraining': 'deepfake_detection.models.pretraining', 'ensembles': 'deepfake_detection.models.ensembles', 'research': 'deepfake_detection.models.research', 'adversarial': 'deepfake_detection.training.adversarial', 'averaging': 'deepfake_detection.training.averaging', 'checkpoints': 'deepfake_detection.training.checkpoints', 'training': 'deepfake_detection.training.trainer', 'optimizers': 'deepfake_detection.training.optimizers', 'schedulers': 'deepfake_detection.training.schedulers', 'precision': 'deepfake_detection.training.precision', 'lr_finder': 'deepfake_detection.training.lr_finder', 'domain_adaptation': 'deepfake_detection.training.domain_adaptation', 'evaluation': 'deepfake_detection.evaluation.evaluator', 'evaluation_extensions': 'deepfake_detection.evaluation.analysis', 'metrics': 'deepfake_detection.evaluation.metrics', 'calibration': 'deepfake_detection.evaluation.calibration', 'explanations': 'deepfake_detection.evaluation.explanations', 'tta': 'deepfake_detection.evaluation.test_time_augmentation', 'api': 'app.backend.api', 'serving': 'deepfake_detection.inference.engine', 'ingestion': 'app.backend.ingestion', '_multipart': 'app.backend.multipart', 'persistence': 'app.backend.persistence', 'client': 'app.streamlit_ui.backend_client', 'paths': 'deepfake_detection.common.paths', 'config': 'deepfake_detection.common.configuration', 'progress': 'deepfake_detection.common.progress', 'presentation': 'deepfake_detection.common.presentation', 'repro': 'deepfake_detection.common.reproducibility', 'runlog': 'deepfake_detection.common.run_log', 'telemetry': 'deepfake_detection.common.telemetry', 'benchmark': 'deepfake_detection.common.benchmark', 'diagnostics': 'deepfake_detection.common.diagnostics'}


def module(test, name):
    try:
        return importlib.import_module(MODULES[name])
    except ModuleNotFoundError as exc:
        if exc.name == MODULES[name]:
            test.fail(name + " is not implemented yet")
        raise


def workspace(test):
    """Temporary outputs stay under the real runs junction; never use OS temp."""
    base = ROOT / "runs" / ".unit-tests"
    base.mkdir(parents=True, exist_ok=True)
    temp = tempfile.TemporaryDirectory(prefix="case-", dir=base)
    path = Path(temp.name)
    assert path.resolve().is_relative_to((ROOT / "runs").resolve())
    test.addCleanup(temp.cleanup)
    (path / "runs").mkdir()
    (path / "crops").mkdir()
    return path
