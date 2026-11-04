from dataclasses import dataclass
import numpy as np
from deepfake_detection.common.configuration import DEFAULTS
from deepfake_detection.data.preprocessing import to_rgb,face_crop_variants


class RetinaFaceDetector:
    def __init__(self,*args,**kwargs):
        raise NotImplementedError('RetinaFace is RESOURCE-PENDING: requires retina-face, explicit local detector weights and an adapter. No packages or weights are downloaded automatically.')


class SCRFDDetector:
    def __init__(self,*args,**kwargs):
        raise NotImplementedError('SCRFD is RESOURCE-PENDING: requires insightface and onnxruntime with explicit local SCRFD weights and an adapter. Offline MTCNN remains available.')


class YOLOFaceDetector:
    def __init__(self,*args,**kwargs):
        raise NotImplementedError('YOLOFace is RESOURCE-PENDING: requires ultralytics and explicit local face-detector weights plus an adapter. No automatic acquisition occurs.')


DETECTOR_BACKENDS={'retinaface':RetinaFaceDetector,'scrfd':SCRFDDetector,'yoloface':YOLOFaceDetector}


@dataclass
class Face:
    box: list
    probability: float
    landmarks: list
    status: str
    crop: object = None
    variant_crops: object = None


class FaceDetector:
    def __init__(self, config=None, *, device=None, backend=None):
        self.config = config or DEFAULTS["preprocessing"]
        selected=self.config.get('detector_backend','mtcnn')
        if selected in DETECTOR_BACKENDS:
            DETECTOR_BACKENDS[selected](self.config,device=device)
        elif selected!='mtcnn':
            raise ValueError('Detector backend must be mtcnn, retinaface, scrfd or yoloface')
        if backend is None:
            import torch
            from facenet_pytorch import MTCNN
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            backend = MTCNN(**self.config["detector"], device=device)
        self.backend = backend

    def detect(self, image):
        image = to_rgb(image)
        if min(image.size) < self.config["min_face_size"]:
            return []
        if image.width * image.height > self.config["max_pixels"]:
            raise ValueError("Image exceeds pixel budget")
        boxes, probs, points = self.backend.detect(image, landmarks=True)
        if boxes is None:
            return []
        if len(boxes) > self.config["max_faces"]:
            raise ValueError("Image exceeds configured detected-face budget")
        result = []
        for index, box in enumerate(boxes):
            box = np.asarray(box, dtype=float)
            prob = float(probs[index])
            if not np.isfinite(box).all() or not np.isfinite(prob):
                continue
            landmark = [] if points is None else np.asarray(points[index], dtype=float).tolist()
            accepted = prob >= self.config["confidence"] and min(box[2] - box[0], box[3] - box[1]) >= self.config["min_face_size"]
            crop = None
            variants=None
            if accepted:
                variants=face_crop_variants(image,box,landmark,self.config)
                crop=next((c for margin,c in variants if margin==self.config['margin_factor']),variants[0][1])
            result.append(Face(box.tolist(), prob, landmark, "accepted" if accepted else "low_quality", crop,variants))
        return result
