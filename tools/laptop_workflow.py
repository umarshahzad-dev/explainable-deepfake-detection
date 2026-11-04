"""Entry point: python -m tools.laptop_workflow (from experiment)."""
import argparse
import base64
import json
import os
from pathlib import Path
import sys

# Absolute script invocation also works, without importing any parent-project code.
EXPERIMENT = Path(__file__).absolute().parents[1]
for import_root in (EXPERIMENT / "src", EXPERIMENT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
from deepfake_detection.common.configuration import load_config
from deepfake_detection.common.paths import Workspace, read_json, write_json, atomic_bytes


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be positive")
    return number


def bounded_steps(value):
    number = positive(value)
    if number > 500:
        raise argparse.ArgumentTypeError("Diagnostics are bounded to 500 microbatches")
    return number


def input_path(value, root=EXPERIMENT):
    path = Path(value)
    return path.absolute() if path.is_absolute() else (root / path).absolute()


def build_parser():
    parser = argparse.ArgumentParser(description="Isolated offline-built laptop deepfake workflow")
    parser.add_argument("--config", default="configs/laptop.json")
    parser.add_argument("--raw-root")
    parser.add_argument("--weights")
    parser.add_argument("--official-splits")
    parser.add_argument("--identity-registry")
    parser.add_argument("--strict-person", action="store_true")
    parser.add_argument("--model", choices=["b0", "xception"], default="b0")
    parser.add_argument("--quiet", action="store_true", help="Disable progress and training status output")
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--limit-files", type=positive, default=2)
    inv.add_argument("--all-files", action="store_true")
    inv.add_argument("--no-decode", action="store_true")
    inv.add_argument("--hash", action="store_true")
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", choices=["ffpp", "dfd", "celebdf"], required=True)
    prep.add_argument("--name")
    prep.add_argument("--limit-videos", type=positive)
    prep.add_argument("--overwrite", action="store_true")
    prep.add_argument("--identity-map")
    prep.add_argument("--require-verified-identities",action="store_true")
    check = sub.add_parser("check-splits")
    check.add_argument("--names", nargs="+")
    ext = sub.add_parser("extract")
    ext.add_argument("--manifest", required=True)
    ext.add_argument("--limit-videos", type=positive)
    ext.add_argument("--max-frames", type=positive)
    ext.add_argument("--attempts-per-segment", type=positive, default=5)
    ext.add_argument("--auto-verify-single-face", action="store_true")
    ext.add_argument("--subject-proximity", type=float, default=0.25, help="Other-face exclusion margin as a fraction of crop side")
    ext.add_argument("--verbose",action="store_true")
    ext.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    verify = sub.add_parser("verify-model")
    verify.add_argument("--forward", action="store_true")
    verify.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    for name in ("profile", "smoke-train"):
        diagnostic = sub.add_parser(name)
        inputs = diagnostic.add_mutually_exclusive_group(required=True)
        inputs.add_argument("--synthetic", action="store_true")
        inputs.add_argument("--manifests", nargs="+")
        diagnostic.add_argument("--steps", type=bounded_steps, default=2)
        diagnostic.add_argument("--warmup", type=int, default=1)
        diagnostic.add_argument("--no-amp", action="store_true", help="Disable AMP for diagnostics only; training configuration is unchanged")
        diagnostic.add_argument("--limit-images", type=positive, default=16)
        diagnostic.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train = sub.add_parser("train")
    finder=sub.add_parser('find-lr')
    finder.add_argument('--run',required=True)
    finder.add_argument('--manifests',nargs='+',required=True)
    finder.add_argument('--steps',type=bounded_steps,default=100)
    finder.add_argument('--start-lr',type=float,default=1e-7)
    finder.add_argument('--end-lr',type=float,default=.01)
    finder.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    bench=sub.add_parser("benchmark")
    bench.add_argument("--manifests",nargs="+",required=True)
    bench.add_argument("--crops",type=positive,default=200)
    bench.add_argument("--batch-size",type=positive)
    bench.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    train.add_argument("--run", required=True)
    train.add_argument("--manifests", nargs="+", required=True)
    train.add_argument("--seed", type=int)
    train.add_argument("--augmentation", choices=["none", "basic", "robust"])
    train.add_argument("--resume", choices=["head", "finetune"])
    train.add_argument("--max-batches", type=positive)
    train.add_argument("--skip-integrity-check", action="store_true", help="Discouraged emergency bypass of sampled PNG checks")
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    calibration = sub.add_parser("calibrate")
    calibration.add_argument("--run", required=True)
    calibration.add_argument("--manifests", nargs="+", required=True)
    calibration.add_argument("--limit-images", type=positive)
    calibration.add_argument("--force",action="store_true")
    calibration.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--manifests", nargs="+", required=True)
    evaluate.add_argument("--raw", action="store_true")
    evaluate.add_argument("--limit-images", type=positive)
    evaluate.add_argument("--limit-videos", type=positive)
    evaluate.add_argument("--max-frames", type=positive)
    evaluate.add_argument("--report-name")
    evaluate.add_argument('--compare-run',help='Explicit prespecified local reference run for paired crop evaluation')
    evaluate.add_argument("--variant", choices=["clean", "jpeg60", "resize_half"], default="clean")
    evaluate.add_argument("--no-plots", action="store_true")
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    explain = sub.add_parser("explain")
    explain.add_argument("--run", required=True)
    explain.add_argument("--image", required=True)
    explain.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    review = sub.add_parser("identity-review")
    audit=sub.add_parser("identity-audit")
    audit.add_argument("--manifests",nargs="+",required=True)
    audit.add_argument("--output")
    review.add_argument("--manifests", nargs="+", required=True)
    review.add_argument("--limit", type=positive, default=200)
    review.add_argument("--embedding-weights")
    deploy = sub.add_parser("deploy")
    deploy.add_argument("--run", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--run", default=os.environ.get("DEEPFAKE_MODEL_ID"))
    serve.add_argument("--port", type=positive, default=os.environ.get("DEEPFAKE_PORT", "8000"))
    sub.add_parser("metric-self-test")
    for command_parser in sub.choices.values():
        command_parser.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS)
    return parser


def device_arg(args):
    import torch
    requested = getattr(args, "device", "auto")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; repair the existing CUDA stack or choose CPU for a small check")
    return requested


def execute(args, cfg, ws):
    command = args.command
    if command=='find-lr':
        from deepfake_detection.data.datasets import crop_rows
        from deepfake_detection.training.lr_finder import find_lr
        return find_lr(args.run,cfg,crop_rows(args.manifests,ws),ws,device=device_arg(args),steps=args.steps,start_lr=args.start_lr,end_lr=args.end_lr)
    if command == "inventory":
        from deepfake_detection.data.inventory import inventory
        return inventory(cfg, ws, limit_files=None if args.all_files else args.limit_files, decode=not args.no_decode, hashes=args.hash)
    if command == "prepare":
        from deepfake_detection.data.manifests import prepare
        return prepare(args.dataset, cfg, ws, limit=args.limit_videos, name=args.name, overwrite=args.overwrite,
                       identity_map=input_path(args.identity_map,ws.root) if args.identity_map else None,require_verified_identities=args.require_verified_identities)
    if command == "check-splits":
        from deepfake_detection.data.manifests import load_video_manifests, check_splits
        result = check_splits(load_video_manifests(ws, args.names))
        write_json(ws.report("split-check") / "split_report.json", result, ws)
        if not result["ok"] or not result["records"]:
            raise ValueError("Split check failed: " + json.dumps(result))
        return result
    if command == "extract":
        from deepfake_detection.data.extraction import extract
        from deepfake_detection.data.face_detector import FaceDetector
        return extract(args.manifest, cfg, ws, limit_videos=args.limit_videos, max_frames=args.max_frames,
                       attempts_per_segment=args.attempts_per_segment,
                       auto_verify_single_face=args.auto_verify_single_face, subject_proximity=args.subject_proximity,verbose=args.verbose,
                       detector=FaceDetector(cfg["preprocessing"], device=device_arg(args)))
    if command == "verify-model":
        from deepfake_detection.models.model_loader import initialize_model, verify_weights
        import torch
        model = initialize_model(cfg, device=device_arg(args)).eval()
        name = cfg["model"]["name"]
        info = verify_weights(cfg["weights"]) if name == "efficientnet_b0.ra_in1k" else verify_weights(cfg["weights"], expected_size=None, expected_sha=None)
        result = {"verified": True, "weights": info, "architecture": name,
                  "classifier_features": model.get_classifier().in_features, "classifier_outputs": model.get_classifier().out_features,
                  "input_size": cfg["model"]["input_size"], "class_map": cfg["model"]["class_map"], "role": "ImageNet initialization; new head is not a trained deepfake classifier"}
        if args.forward:
            with torch.no_grad():
                output = model(torch.zeros(1, 3, cfg["model"]["input_size"], cfg["model"]["input_size"], device=device_arg(args)))
            result["output_shape"] = list(output.shape)
            result["finite_output"] = bool(torch.isfinite(output).all())
        write_json(ws.report("verify-model") / "verification.json", result, ws)
        return result
    if command in {"profile", "smoke-train"}:
        from deepfake_detection.common.diagnostics import diagnostic
        return diagnostic(cfg, ws, synthetic=args.synthetic, manifests=args.manifests, steps=args.steps, warmup=args.warmup,
                          limit_images=args.limit_images, kind=command, device=device_arg(args), no_amp=args.no_amp)
    if command == "train":
        from deepfake_detection.data.datasets import crop_rows
        from deepfake_detection.training.trainer import run_training
        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.augmentation:
            cfg["training"]["augmentation"] = args.augmentation
        return run_training(args.run, cfg, crop_rows(args.manifests, ws), ws, device=device_arg(args), resume=args.resume, max_batches=args.max_batches,skip_integrity_check=args.skip_integrity_check)
    if command=="benchmark":
        from deepfake_detection.common.benchmark import benchmark
        return benchmark(cfg,args.manifests,ws,crops=args.crops,device=device_arg(args),batch_size=args.batch_size)
    if command == "calibrate":
        from deepfake_detection.evaluation.evaluator import calibrate
        return calibrate(args.run, args.manifests, ws, device=device_arg(args), limit=args.limit_images,force=args.force)
    if command == "evaluate":
        from deepfake_detection.evaluation.evaluator import evaluate, evaluate_raw
        if args.raw:
            if args.compare_run: raise ValueError('Paired comparison currently requires crop-mode evaluation')
            return evaluate_raw(args.run, args.manifests, ws, device=device_arg(args), limit_videos=args.limit_videos,
                                max_frames=args.max_frames, report_name=args.report_name, plots=not args.no_plots)
        return evaluate(args.run, args.manifests, ws, device=device_arg(args), limit=args.limit_images,
                        report_name=args.report_name, plots=not args.no_plots, variant=args.variant,compare_run=args.compare_run)
    if command == "explain":
        from deepfake_detection.inference.engine import InferenceEngine, decode_image
        path = input_path(args.image, ws.root)
        image = decode_image(path.read_bytes())
        result = InferenceEngine.from_run(args.run, ws, device=device_arg(args)).analyze(image, explain=True)
        output = ws.report(args.run + "-explanations")
        for face in result["faces"]:
            for key, suffix in (("heatmap_data_url", "cam"), ("overlay_data_url", "overlay")):
                if face.get(key):
                    file = output / f"face-{face['face_index']}-{suffix}.png"
                    atomic_bytes(file, base64.b64decode(face[key].split(",", 1)[1]), ws)
                    face[suffix + "_path"] = str(file)
        write_json(output / "analysis.json", result, ws)
        return result
    if command == "identity-review":
        from deepfake_detection.data.datasets import crop_rows
        from deepfake_detection.data.review import contact_review
        return contact_review(crop_rows(args.manifests, ws), ws, limit=args.limit,
                              embedding_weights=input_path(args.embedding_weights, ws.root) if args.embedding_weights else None, seed=cfg["seed"])
    if command=="identity-audit":
        from deepfake_detection.data.identity import identity_audit
        return identity_audit(args.manifests,ws,output=args.output)
    if command == "deploy":
        from deepfake_detection.models.model_loader import load_run_model
        from deepfake_detection.models.registry import resolve_model_run
        _, _, metadata = load_run_model(resolve_model_run(args.run, ws), workspace=ws)
        result = {"run_id": args.run, "weights_sha256": metadata["weights_sha256"]}
        write_json(ws.runs / "deployment.json", result, ws)
        return result
    if command == "serve":
        import uvicorn
        from app.backend.api import create_app
        if args.port > 65535:
            raise ValueError("Port must be <=65535")
        uvicorn.run(create_app(workspace=ws, run=args.run), host="127.0.0.1", port=args.port, workers=1)
        return {"status": "server_stopped"}
    if command == "metric-self-test":
        from deepfake_detection.evaluation.metrics import metric_fixture
        result = metric_fixture()
        write_json(ws.report("metric-self-test") / "fixture.json", result, ws)
        return result
    raise ValueError("Unknown command")


def _invoke(args, ws):
    overrides = {key: str(input_path(value, ws.root)) for key in ("raw_root", "weights", "official_splits", "identity_registry") if (value := getattr(args, key))}
    if args.strict_person:
        overrides["identity_policy"] = "strict-person"
    if args.model == "xception":
        if not args.weights:
            print("ERROR: Fresh Xception needs an explicit offline ImageNet --weights file; old checkpoints are forbidden", file=sys.stderr)
            return 2
        overrides["model"] = {"name": "legacy_xception.tf_in1k", "input_size": 299, "mean": [0.5] * 3, "std": [0.5] * 3}
    try:
        # Programmatic fixture workspaces use defaults; production reads the checked-in experiment config.
        path = args.config
        if path == "configs/laptop.json" and not (ws.root / path).exists():
            path = None
        cfg = load_config(path, root=ws.root, overrides=overrides)
        from deepfake_detection.common.progress import progress_options
        with progress_options(quiet=args.quiet):
            from deepfake_detection.common.presentation import first_banner,print_summary
            first_banner(args.config,device_arg(args),ws)
            result = execute(args, cfg, ws)
            if args.command in {"extract","train","evaluate","calibrate","prepare","benchmark","identity-audit","find-lr"}:
                print_summary(args.command,result)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
        if args.command=="identity-audit": return result["exit_code"]
        if args.command == "extract" and result.get("failed_videos", 0):
            print("ERROR: Some videos failed; inspect extraction_report.json and repair before training", file=sys.stderr)
            return 2
        if args.command == "prepare" and not result.get("admitted", 0):
            print("ERROR: No admitted videos; inspect quarantine and review policy", file=sys.stderr)
            return 2
        return 0
    except (OSError, ValueError, RuntimeError, FloatingPointError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


def main(argv=None, *, workspace=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    ws = workspace or Workspace()
    from deepfake_detection.common.run_log import RunLog, invocation_log_path
    try:
        path = invocation_log_path(args, ws)
        if path is None:
            return _invoke(args, ws)
        with RunLog(path, [sys.executable, "-m", "tools.laptop_workflow", *arguments], ws) as log:
            code = _invoke(args, ws)
            log.exit_status = code
            return code
    except OSError as exc:
        print(f"ERROR: Cannot persist invocation log: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
