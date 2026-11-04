"""Windows terminal progress, redirected log-friendly updates, and test silence."""
from contextlib import contextmanager
from contextvars import ContextVar
import sys
import time
import os

_quiet = ContextVar("workflow_quiet", default=False)


def is_test_mode():
    """Detect a test invocation, never merely an imported test-library module."""
    marker = os.environ.get("DEEPFAKE_EXPERIMENT_TEST_MODE")
    if marker is not None:
        return marker.strip().lower() in {"1", "true", "yes", "on"}
    launcher = (sys.argv[0] if sys.argv else "").replace("\\", "/").lower()
    parts = launcher.split("/")
    return "unittest" in parts or parts[-1] == "unittest.exe"


@contextmanager
def progress_options(quiet=False):
    token = _quiet.set(quiet)
    try:
        yield
    finally:
        _quiet.reset(token)


def duration(seconds):
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds//3600}h{seconds%3600//60:02d}m"
    return f"{seconds//60}m" if seconds >= 60 else f"{seconds}s"


def console(message):
    if not _quiet.get() and not is_test_mode():
        print(message, flush=True)


class Progress:
    def __init__(self, label, total, *, unit="", stream=None, enabled=None, interval=1, clock=time.monotonic, suffix=""):
        self.stream = stream or sys.stdout
        self.enabled = (not is_test_mode() if enabled is None else enabled) and not _quiet.get()
        self.tty = bool(self.stream.isatty())
        self.interval = interval if self.tty else max(interval, 10)
        self.clock, self.started = clock, clock()
        self.last = None
        self.width = 0
        self.label, self.total, self.unit, self.suffix = label, total, unit, suffix
        self.update(0,force=True)

    def update(self, current, detail="", *, force=False):
        if not self.enabled:
            return
        now = self.clock()
        if not force and self.last is not None and now-self.last < self.interval:
            return
        elapsed = now - self.started
        eta = duration(elapsed * max(0, self.total-current) / current) if current else "pending"
        message = f"{self.label} {current}/{self.total}" + (f" {self.unit}" if self.unit else "")
        if detail:
            message += " · " + detail
        message += self.suffix + f" · elapsed {duration(elapsed)} · ETA {eta}"
        if self.tty:
            self.stream.write("\r" + message.ljust(self.width))
            self.width = len(message)
        else:
            self.stream.write(message + "\n")
        self.stream.flush()
        self.last = now

    def finish(self, current, detail=""):
        self.update(current, detail, force=True)
        if self.enabled and self.tty:
            self.stream.write("\n")
            self.stream.flush()
