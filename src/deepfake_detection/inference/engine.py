import base64
from io import BytesIO
import threading
import time
import numpy as np
import torch
from PIL import Image
from deepfake_detection.data.face_detector import FaceDetector
from deepfake_detection.evaluation.evaluator import calibration_for
from deepfake_detection.evaluation.explanations import grad_cam
from deepfake_detection.evaluation.metrics import real_probability
from deepfake_detection.evaluation.calibration import apply_calibration
from deepfake_detection.models.model_loader import load_run_model
from deepfake_detection.models.registry import resolve_model_run
from deepfake_detection.common.paths import Workspace,read_json
from deepfake_detection.data.preprocessing import model_tensor, to_rgb, validate_image, overlay_heatmap

MAX_BYTES = 10 * 1024 * 1024


def decode_image(data, *, max_pixels=8000000):
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Image is empty or exceeds the 10 MiB byte limit")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPEG and PNG still images are supported")
            # Check dimensions/animation before full decompression/conversion.
            if image.width * image.height > max_pixels or getattr(image, "is_animated", False):
                raise ValueError("Animated or oversized image")
            return validate_image(image, max_pixels)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Corrupt or unsupported image") from exc


def png_data_url(image):
    stream = BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


class InferenceEngine:
    def __init__(self, model, cfg, metadata, *, detector=None, calibration=None, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.cfg, self.metadata = cfg, metadata
        self.detector = detector or FaceDetector(cfg["preprocessing"], device=str(self.device))
        self.calibration = calibration or {"temperature": 1.0, "threshold": 0.5, "applied": False}
        self.lock = threading.Lock()

    @classmethod
    def from_run(cls, name, workspace=None, *, device=None,allow_ensemble=True):
        ws = workspace or Workspace()
        run_path = resolve_model_run(name, ws)
        if allow_ensemble:
            saved=read_json(run_path/'config.json')
            selection=saved.get('ensemble',{})
            if selection.get('enabled'):
                from deepfake_detection.models.ensembles import EnsembleScorer
                names=selection.get('runs',[])
                if len(names)<2 or len(set(names))!=len(names): raise ValueError('Enabled ensemble requires at least two explicit distinct local runs')
                return EnsembleScorer.from_runs(names,selection.get('weights',saved.get('inference',{}).get('ensemble_weights')),workspace=ws,device=selection.get('device',device or 'cpu'),threshold=selection.get('threshold',.5),fusion=selection.get('fusion','weighted_average'))
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, cfg, metadata = load_run_model(run_path, workspace=ws, device=device)
        return cls(model, cfg, metadata, device=device, calibration=calibration_for(run_path, metadata,cfg))

    def analyze(self, image, *, explain=False):
        with self.lock:
            started = time.perf_counter()
            image = validate_image(image, self.cfg["preprocessing"]["max_pixels"])
            faces = self.detector.detect(image)
            results = []
            for index, face in enumerate(faces):
                item = {"face_index": index, "box": face.box, "status": face.status,
                        "detection_probability": face.probability, "label": None,
                        "fake_score": None, "real_score": None, "confidence": None,
                        "heatmap_data_url": None, "overlay_data_url": None}
                if face.status == "accepted":
                    tensor = model_tensor(face.crop.image, self.cfg["model"]).unsqueeze(0).to(self.device)
                    from deepfake_detection.evaluation.test_time_augmentation import inference_augmented,inference_score
                    if inference_augmented(self.cfg):
                        prediction=inference_score(self.model,face.crop.image,self.cfg,device=self.device)
                        logits=np.asarray([prediction['logit']])
                        item['tta_views']=prediction['views']; item['explanation_scope']='unaugmented view'
                        if 'mc_dropout_real_std' in prediction: item['mc_dropout_real_std']=prediction['mc_dropout_real_std']; item['mc_samples']=prediction['mc_samples']
                    else:
                        with torch.no_grad(): logits = self.model(tensor).float().cpu().reshape(-1).numpy()
                    if len(logits) != 1:
                        raise ValueError("The selected model is not a one-logit classifier")
                    real = float(apply_calibration(real_probability(logits),self.calibration)[0])
                    fake = 1 - real
                    label = 0 if fake >= self.calibration["threshold"] else 1
                    item.update({"label": "FAKE" if label == 0 else "REAL", "fake_score": fake, "real_score": real,
                                 "confidence": fake if label == 0 else real, "uncertain": abs(fake - self.calibration["threshold"]) < 0.05,
                                 "geometry": face.crop.geometry.as_dict(), "landmarks": face.landmarks})
                    if explain:
                        cam = grad_cam(self.model, tensor, target_class=label)
                        item["heatmap_data_url"] = png_data_url(Image.fromarray(np.uint8(cam * 255)))
                        overlay = overlay_heatmap(image, face.crop.geometry, cam)
                        overlay.thumbnail((1024, 1024), Image.Resampling.BICUBIC)
                        item["overlay_data_url"] = png_data_url(overlay)
                        item["explanation_informative"] = bool(np.any(cam > 0))
                results.append(item)
            preview = image.copy()
            preview.thumbnail((1024, 1024), Image.Resampling.BICUBIC)
            return {"run_id": self.metadata["run_id"], "weights_sha256": self.metadata["weights_sha256"],
                    "preprocessing_hash": self.metadata["preprocessing_hash"], "class_map": {"fake": 0, "real": 1},
                    "calibration_applied": bool(self.calibration.get("applied")), "temperature": self.calibration["temperature"], "threshold": self.calibration["threshold"],
                    "training_pilot": bool(self.metadata.get("pilot", False)), "faces": results,
                    "status": "no_faces" if not faces else "completed", "elapsed_seconds": time.perf_counter() - started,
                    "image": {"width": image.width, "height": image.height, "preview_data_url": png_data_url(preview)},
                    "explanation_note": "Attribution shows model influence, not proof of manipulation."}
