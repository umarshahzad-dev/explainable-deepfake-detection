"""Offline environment capture and sampled process VRAM with honest WDDM limits."""
import importlib.metadata
import os
import platform
import shutil
import subprocess
import threading


def environment_snapshot():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "torchvision", "timm", "facenet-pytorch", "numpy", "Pillow", "opencv-python", "scikit-learn", "safetensors"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unavailable"
    return result


def parse_process_memory(text, pid):
    values = []
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            try:
                values.append(float(fields[1]))
            except ValueError:
                return None
    return sum(values) if values else None


class MemoryMonitor:
    def __init__(self, device="cpu"):
        self.device = str(device)
        self.peak = None
        self.event = threading.Event()
        self.thread = None
        self.executable = shutil.which("nvidia-smi") if self.device.startswith("cuda") else None

    def _sample(self):
        if not self.executable:
            return
        try:
            result = subprocess.run([self.executable, "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=2,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            value = parse_process_memory(result.stdout, os.getpid()) if result.returncode == 0 else None
            if value is not None:
                self.peak = max(self.peak or 0, value)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _poll(self):
        while not self.event.is_set():
            self._sample()
            self.event.wait(0.5)

    def start(self):
        if self.executable:
            self.thread = threading.Thread(target=self._poll, daemon=True)
            self.thread.start()
        return self

    def stop(self):
        self.event.set()
        if self.thread:
            self.thread.join(timeout=3)
        return self.report()

    def report(self):
        import torch
        cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        return {"peak_process_mib": self.peak,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / 2**20 if cuda else None,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(self.device) / 2**20 if cuda else None,
                "limitation": "Process peak is sampled every 0.5s; unavailable on CPU or when nvidia-smi/WDDM withholds per-process memory. Allocator peaks exclude driver/other libraries."}

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()
