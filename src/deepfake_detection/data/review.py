from collections import defaultdict
from io import BytesIO
from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw
from deepfake_detection.common.paths import Workspace, atomic_bytes, digest_file, write_json
from deepfake_detection.common.reproducibility import seed_all


def contact_review(rows, workspace=None, *, limit=200, embedding_weights=None, seed=42):
    """Candidate assistance only; never writes a verified person or auto-merges IDs."""
    ws = workspace or Workspace()
    output = ws.report("identity-review")
    groups = defaultdict(list)
    for row in rows:
        if row["label"] == 1:
            for parent in row["parents"]:
                groups[parent].append(row)
    parents = sorted(groups)[:limit]
    if not parents:
        raise ValueError("Extract some original-real crops before building identity review sheets")
    sheets = []
    for offset in range(0, len(parents), 20):
        page = parents[offset:offset + 20]
        sheet = Image.new("RGB", (800, 4 * 190), "white")
        draw = ImageDraw.Draw(sheet)
        for index, parent in enumerate(page):
            path = ws.output(groups[parent][0]["crop_path"])
            with Image.open(path) as image:
                thumbnail = image.convert("RGB").resize((150, 150), Image.Resampling.BICUBIC)
            x, y = (index % 5) * 160, (index // 5) * 190
            sheet.paste(thumbnail, (x, y))
            draw.text((x, y + 153), parent, fill="black")
        path = output / f"contacts-{offset // 20:03d}.png"
        stream = BytesIO()
        sheet.save(stream, format="PNG")
        atomic_bytes(path, stream.getvalue(), ws)
        sheets.append(str(path))
    template_path = output / "identity_registry_template.json"
    subjects = {}
    for row in rows:
        if row["label"] == 0:
            subjects.setdefault(row.get("video_id", row["sample_id"]), row)
    subject_keys = sorted(subjects)[:limit]
    subject_sheets = []
    for offset in range(0, len(subject_keys), 20):
        sheet = Image.new("RGB", (800, 4 * 210), "white")
        draw = ImageDraw.Draw(sheet)
        for index, key in enumerate(subject_keys[offset:offset + 20]):
            row = subjects[key]
            with Image.open(ws.output(row["crop_path"])) as image:
                thumb = image.convert("RGB").resize((150, 150), Image.Resampling.BICUBIC)
            x, y = index % 5 * 160, index // 5 * 210
            sheet.paste(thumb, (x, y))
            draw.text((x, y + 153), key[:23] + "\n" + key[23:46], fill="black")
        path = output / f"subjects-{offset // 20:03d}.png"
        stream = BytesIO()
        sheet.save(stream, format="PNG")
        atomic_bytes(path, stream.getvalue(), ws)
        subject_sheets.append(str(path))
    if template_path.exists():
        # Never overwrite a file someone may already have annotated.
        template_path = output / "identity_registry_template_new.json"
    write_json(template_path, {"identities": {parent: {"person_id": None, "verified": False} for parent in parents},
                              "subjects": {key: {"verified": False, "evidence": None, "raw_path": subjects[key].get("raw_path"), "parents": subjects[key]["parents"]} for key in subject_keys},
                              "permissions": {},
                              "note": "Copy to a new registry, manually verify aliases, then pass --identity-registry. Namespaces alone do not prove different people."}, ws)
    result = {"parents": len(parents), "sheets": sheets, "subject_sheets": subject_sheets, "template_path": str(template_path), "embedding_candidates": False,
              "limitation": "Manual person review required; no identities auto-verified."}
    if embedding_weights:
        embeddings = _offline_face_embeddings(groups, parents, embedding_weights, seed)
        candidates = []
        for i, left in enumerate(parents):
            for right in parents[i + 1:]:
                score = float(np.dot(embeddings[left], embeddings[right]))
                if score >= 0.8:
                    candidates.append({"left": left, "right": right, "cosine": score})
        candidates.sort(key=lambda row: -row["cosine"])
        stream = __import__("io").StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=["left", "right", "cosine"])
        writer.writeheader()
        writer.writerows(candidates)
        atomic_bytes(output / "embedding_candidates.csv", stream.getvalue().encode(), ws)
        result.update({"embedding_candidates": True, "candidate_pairs": len(candidates), "embedding_weights_sha256": digest_file(embedding_weights),
                       "embedding_limitation": "Forensic crops differ from embedding training alignment; cosine matches are review suggestions, not identity guarantees."})
    write_json(output / "review_report.json", result, ws)
    return result


def _offline_face_embeddings(groups, parents, weights_path, seed):
    import torch
    from facenet_pytorch import InceptionResnetV1
    from safetensors.torch import load_file
    seed_all(seed)
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError("An explicit local InceptionResnetV1 weight file is required; downloading is disabled")
    model = InceptionResnetV1(pretrained=None, classify=False)
    state = load_file(str(path)) if path.suffix == ".safetensors" else torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith("logits.")}, strict=True)
    model.eval()
    result = {}
    with torch.no_grad():
        for parent in parents:
            tensors = []
            for row in groups[parent][:3]:
                with Image.open(row["crop_path"]) as image:
                    values = np.asarray(image.convert("RGB").resize((160, 160), Image.Resampling.BILINEAR), dtype=np.float32)
                tensors.append(torch.from_numpy(values.transpose(2, 0, 1).copy()).sub(127.5).div(128))
            value = model(torch.stack(tensors)).mean(0)
            value = torch.nn.functional.normalize(value, dim=0)
            result[parent] = value.numpy()
    return result
