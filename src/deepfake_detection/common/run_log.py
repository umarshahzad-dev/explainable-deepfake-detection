"""Append-only timestamped Python stdout/stderr tee for explicit CLI invocations."""
from datetime import datetime, timezone
import subprocess
import sys
import threading
import traceback
from contextvars import ContextVar
from deepfake_detection.common.paths import Workspace
_active=ContextVar("active_run_log",default=None)


def audit_message(channel,message):
    log=_active.get()
    if log is not None: log.line(channel,message)


class _Tee:
    def __init__(self, original, log, channel):
        self.original, self.log, self.channel = original, log, channel
        self.pending = ""

    def write(self, text):
        self.original.write(text)
        self.pending += text.replace("\r", "\n")
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line:
                self.log.line(self.channel, line)
        return len(text)

    def flush(self):
        self.original.flush()
        if self.pending:
            self.log.line(self.channel, self.pending)
            self.pending = ""

    def isatty(self):
        return self.original.isatty()

    def __getattr__(self, name):
        return getattr(self.original, name)


class RunLog:
    def __init__(self, path, command, workspace=None):
        self.path = (workspace or Workspace()).output(path)
        self.command = subprocess.list2cmdline([str(value) for value in command])
        self.exit_status = 0
        self.lock = threading.Lock()

    def line(self, channel, message):
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.lock:
            for line in str(message).splitlines():
                self.file.write(f"[{timestamp}] [{channel}] {line}\n")
            self.file.flush()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a", encoding="utf-8", newline="\n")
        self.line("RUN", "START command=" + self.command)
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self.stdout, self, "STDOUT")
        sys.stderr = _Tee(self.stderr, self, "STDERR")
        self.token=_active.set(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            if exc is not None:
                self.exit_status = 130 if isinstance(exc, KeyboardInterrupt) else (exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1)
                self.line("STDERR", "".join(traceback.format_exception(exc_type, exc, tb)))
            self.line("RUN", f"END exit_status={self.exit_status}")
        finally:
            sys.stdout, sys.stderr = self.stdout, self.stderr
            _active.reset(self.token)
            self.file.close()


def invocation_log_path(args, workspace):
    if args.command in {"train", "calibrate", "find-lr"}:
        return workspace.run(args.run) / "run.log"
    if args.command == "extract":
        return workspace.run(args.manifest) / "run.log"
    if args.command == "prepare":
        return workspace.run(args.name or args.dataset)/"run.log"
    if args.command=="benchmark": return workspace.report("benchmark")/"run.log"
    if args.command=="identity-audit":
        return workspace.output(args.output,explicit=True)/"run.log" if args.output else workspace.report("identity-audit")/"run.log"
    if args.command == "evaluate":
        suffix = "-raw-evaluation" if args.raw else "-evaluation"
        return workspace.report(args.report_name or args.run + suffix) / "run.log"
    return None
