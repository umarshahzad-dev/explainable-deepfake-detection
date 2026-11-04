"""Lexical experiment paths with resolved containment checks for junction safety."""
import hashlib
import json
import os
import re
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class Workspace:
    def __init__(self, root=ROOT):
        self.root = Path(root).absolute()
        self.crops = self.root / "crops"
        self.runs = self.root / "runs"

    def output(self, path, *, explicit=False):
        path = Path(path)
        if not path.is_absolute():
            path = self.root / path
        path = path.absolute()
        resolved = path.resolve()
        anchors = (self.root.resolve(), self.crops.resolve(), self.runs.resolve())
        # Explicit flags choose a destination, not permission to edit the parent repository.
        if not any(resolved.is_relative_to(a) for a in anchors):
            raise ValueError(f"Output escapes experiment/junctions: {path}")
        return path

    def _name(self, name):
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_./-]*", name) or ".." in name:
            raise ValueError("Use a safe run/report name or subfolder path")
        return name

    def run(self, name):
        return self.output(self.runs / self._name(name))

    def report(self, name):
        return self.output(self.runs / "reports" / self._name(name))

    def manifest(self, name):
        return self.output(self.runs / "manifests" / self._name(name))


def digest_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_bytes(path, data, workspace=None, *, explicit=False):
    path = (workspace or Workspace()).output(path, explicit=explicit)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return path


def write_json(path, value, workspace=None, *, explicit=False):
    return atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"), workspace, explicit=explicit)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_jsonl(path, rows, workspace=None):
    data = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    return atomic_bytes(path, data.encode("utf-8"), workspace)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]
