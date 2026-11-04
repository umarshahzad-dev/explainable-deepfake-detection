"""One-process API. Constructing/importing it never trains, extracts, or downloads."""
import hashlib
import json
import threading
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from app.backend.ingestion import fetch_public_image
from deepfake_detection.common.paths import Workspace, read_json
from app.backend.persistence import HistoryStore
from deepfake_detection.inference.engine import InferenceEngine, MAX_BYTES, decode_image
from deepfake_detection.models.registry import resolve_model_run

logger = logging.getLogger(__name__)


def multipart_available():
    try:
        try:
            from python_multipart.multipart import parse_options_header
        except ImportError:
            from multipart.multipart import parse_options_header
        return True
    except ImportError:
        return False


async def multipart_image(headers, body, explain):
    from starlette.formparsers import MultiPartParser, MultiPartException
    from starlette.datastructures import UploadFile
    async def chunks():
        yield bytes(body)
    parser = MultiPartParser(headers, chunks(), max_files=1, max_fields=1, max_part_size=1024)
    # The entire envelope was already bounded to MAX_BYTES. Never spill uploads to OS temp.
    parser.spool_max_size = MAX_BYTES + 1
    parser.max_file_size = MAX_BYTES + 1
    form = None
    try:
        form = await parser.parse()
        items = list(form.multi_items())
        upload = form.get("file")
        if not isinstance(upload, UploadFile) or sum(isinstance(value, UploadFile) for _, value in items) != 1:
            raise ValueError("Multipart requires exactly one image upload named file")
        if any(key not in {"file", "explain"} for key, _ in items):
            raise ValueError("Unknown multipart field")
        if "explain" in form:
            value = form["explain"]
            if not isinstance(value, str) or value.lower() not in {"true", "false"}:
                raise ValueError("Multipart explain must be true or false")
            explain = value.lower() == "true"
        return await upload.read(MAX_BYTES + 1), explain
    except MultiPartException as exc:
        raise ValueError("Malformed multipart input: " + str(exc)) from exc
    finally:
        if form is not None:
            await form.close()
        else:
            for file in parser._files_to_close_on_error:
                file.close()


def create_app(*, engine=None, workspace=None, run=None, fetcher=fetch_public_image):
    ws = workspace or Workspace()
    app = FastAPI(title="Offline-built Deepfake Experiment", version="1.0.0")
    app.state.multipart_enabled = multipart_available()
    if not app.state.multipart_enabled:
        logger.warning("Optional python-multipart unavailable; using the built-in parser alongside raw-body uploads and JSON URLs")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    cache = {"engine": engine, "key": None}
    startup_lock = threading.Lock()
    admission = threading.BoundedSemaphore(4)
    store = HistoryStore(ws)
    def current_engine():
        if engine is not None:
            return engine
        selected = run
        if selected is None:
            deployment = ws.runs / "deployment.json"
            if not deployment.exists():
                raise FileNotFoundError("No deployment selected; use the CLI deploy command after training")
            selected = read_json(deployment)["run_id"]
        folder = resolve_model_run(selected, ws)
        files = [folder / "best.safetensors", folder / "best_metadata.json", folder / "config.json", folder / "calibration.json"]
        key = (selected, tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else (str(p), None, None) for p in files))
        with startup_lock:
            if cache["key"] != key:
                cache["engine"] = InferenceEngine.from_run(selected, ws)
                cache["key"] = key
        return cache["engine"]

    @app.get("/health")
    def health():
        try:
            value = current_engine()
            return {"ready": True, "run_id": value.metadata["run_id"], "weights_sha256": value.metadata["weights_sha256"]}
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(503, detail=str(exc)) from exc

    @app.get("/history")
    def history(limit: int = 20):
        return store.recent(limit)

    @app.post("/analyze")
    async def analyze(request: Request, explain: bool = False):
        if not admission.acquire(blocking=False):
            raise HTTPException(429, detail="Analysis queue is full; retry later")
        try:
            data = bytearray()
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > MAX_BYTES:
                    raise HTTPException(413, detail="Request exceeds 10 MiB")
            source = "upload"
            content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type == "multipart/form-data":
                if not app.state.multipart_enabled:
                    from app.backend.multipart import parse_multipart
                    parts=await run_in_threadpool(parse_multipart,request.headers["content-type"],bytes(data),max_bytes=MAX_BYTES)
                    files=[part for part in parts if part["name"]=="file" and part["filename"] is not None]
                    if len(files)!=1: raise ValueError("Multipart requires exactly one uploaded image named file")
                    for part in parts:
                        if part is not files[0]: logger.warning("Ignored multipart field %r",part["name"])
                    data=files[0]["data"]
                else:
                    data, explain = await multipart_image(request.headers, data, explain)
            elif content_type == "application/json":
                payload = json.loads(data)
                if not isinstance(payload, dict) or not isinstance(payload.get("url"), str):
                    raise ValueError("JSON input requires one image URL")
                if "explain" in payload and not isinstance(payload["explain"], bool):
                    raise ValueError("explain must be a boolean")
                explain = payload.get("explain", explain)
                data = await run_in_threadpool(fetcher, payload["url"])
                source = "url"
            image = decode_image(bytes(data))
            try:
                predictor = await run_in_threadpool(current_engine)
            except (OSError, ValueError, RuntimeError) as exc:
                raise HTTPException(503, detail="Model is not ready: " + str(exc)) from exc
            result = await run_in_threadpool(predictor.analyze, image, explain=explain)
            result["analysis_id"] = await run_in_threadpool(store.record, hashlib.sha256(data).hexdigest(), source, result)
            return result
        except HTTPException:
            raise
        except (ValueError, OSError, TimeoutError, json.JSONDecodeError) as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, detail="Inference failed; check GPU availability and server logs") from exc
        finally:
            admission.release()
    return app
