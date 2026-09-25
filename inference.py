"""Command-line MimicMotion inference. Model dependencies load only after CLI validation."""
import argparse
import hashlib
import json
import logging
import math
import os
import platform
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

logger = logging.getLogger(__name__)


def preprocess(video_path, image_path, resolution=576, sample_stride=2, processor=None, min_frames=2):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): input video pose path
        image_path (str): reference image path
        resolution (int, optional):  Defaults to 576.
        sample_stride (int, optional): Defaults to 2.
    """
    import numpy as np
    import torch
    from torchvision.datasets.folder import pil_loader
    from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
    from constants import ASPECT_RATIO
    from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose

    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels) # (c, h, w)
    h, w = image_pixels.shape[-2:]
    ############################ compute target h/w according to original aspect ratio ###############################
    if h>w:
        w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    else:
        w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution
    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()
    ##################################### get image&video pose value #################################################
    image_pose = get_image_pose(image_pixels, processor=processor)
    video_pose = get_video_pose(video_path, image_pixels, sample_stride=sample_stride, processor=processor, min_frames=min_frames)
    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    return torch.from_numpy(pose_pixels.copy()) / 127.5 - 1, torch.from_numpy(image_pixels) / 127.5 - 1

def run_pipeline(pipeline, image_pixels, pose_pixels, device, task_config):
    import torch
    from torchvision.transforms.functional import to_pil_image

    pose_pixels = pose_pixels.to(dtype=next(pipeline.pose_net.parameters()).dtype)
    image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
    generator = torch.Generator(device=device)
    generator.manual_seed(task_config.seed)
    frames = pipeline(
        image_pixels, image_pose=pose_pixels, num_frames=pose_pixels.size(0),
        tile_size=task_config.num_frames, tile_overlap=task_config.frames_overlap,
        height=pose_pixels.shape[-2], width=pose_pixels.shape[-1], fps=task_config.conditioning_fps,
        noise_aug_strength=task_config.noise_aug_strength, num_inference_steps=task_config.num_inference_steps,
        generator=generator, min_guidance_scale=task_config.guidance_scale, 
        max_guidance_scale=task_config.guidance_scale, decode_chunk_size=task_config.decode_chunk_size, output_type="pt", device=device
    ).frames.cpu()
    video_frames = (frames * 255.0).to(torch.uint8)

    for vid_idx in range(video_frames.shape[0]):
        # deprecated first frame because of ref image
        _video_frames = video_frames[vid_idx, 1:]

    return _video_frames


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a video from a reference image and pose video (no YAML required).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_model_path", default="stabilityai/stable-video-diffusion-img2vid-xt-1-1",
                        help="Base SVD model directory or Hugging Face model ID")
    parser.add_argument("--ckpt_path", default="models/MimicMotion_1-1.pth",
                        help="MimicMotion checkpoint path")
    parser.add_argument("--ref_video_path", required=True, nargs="+", help="Pose video paths; paired with image paths in order")
    parser.add_argument("--ref_image_path", required=True, nargs="+", help="Image paths; supply one to reuse across videos")
    parser.add_argument("--num_frames", type=int, default=72, help="Temporal tile size, not output video length")
    parser.add_argument("--resolution", type=int, default=576, help="Target short-side resolution (multiple of 64)")
    parser.add_argument("--frames_overlap", type=int, default=6, help="Overlap between temporal tiles")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Denoising steps")
    parser.add_argument("--noise_aug_strength", type=float, default=0.0, help="Noise augmentation strength")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="Guidance scale")
    parser.add_argument("--sample_stride", type=int, default=2, help="Pose video sampling stride")
    parser.add_argument("--fps", type=int, default=15, help="Output video frame rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_file", type=str, default=None, help="Log file path")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Output directory")
    parser.add_argument("--no_use_float16", action="store_true",
                        help="Compatibility alias for --dtype float32")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("float16", "float32"), default=None,
                        help="Default: float16 on CUDA, float32 on CPU")
    parser.add_argument("--decode_chunk_size", type=int, default=8, help="Frames decoded per chunk")
    parser.add_argument("--conditioning_fps", type=int, default=7, help="Model conditioning FPS, separate from output FPS")
    parser.add_argument("--output_file", help="Exact output path: .mov for transparency, otherwise .mp4; single input only")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output and metadata")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue other tasks after a failure; exit remains nonzero")
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    from mimicmotion.utils.replacement_cli import add_arguments, validate_arguments
    add_arguments(parser)
    parser.set_defaults(sample_stride=None, fps=None)
    args = parser.parse_args(argv)
    if args.sample_stride is None:
        args.sample_stride = 1 if args.mode == 'replace' else 2
    if args.fps is None and args.mode == 'generate':
        args.fps = 15
    validate_arguments(args, parser)
    for name in ("num_frames", "resolution", "num_inference_steps", "sample_stride", "fps", "decode_chunk_size", "conditioning_fps"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name} must be greater than zero")
    if args.resolution % 64:
        parser.error("--resolution must be a multiple of 64")
    if not 0 <= args.frames_overlap < args.num_frames:
        parser.error("--frames_overlap must satisfy 0 <= frames_overlap < num_frames")
    for name in ("noise_aug_strength", "guidance_scale"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name} must be finite and non-negative")
    if args.num_frames < 2:
        parser.error("--num_frames must be at least 2")
    if args.guidance_scale <= 1:
        parser.error("--guidance_scale must exceed 1 for this pipeline's guided inference")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be in [0, 2**63)")
    if args.no_use_float16:
        if args.dtype == "float16":
            parser.error("--no_use_float16 conflicts with --dtype float16")
        args.dtype = "float32"
    if len(args.ref_image_path) not in (1, len(args.ref_video_path)):
        parser.error("Provide one reference image or one image per video")
    if args.output_file and len(args.ref_video_path) != 1:
        parser.error("--output_file requires exactly one video")
    extension = ".mov" if args.transparent_background else ".mp4"
    if args.output_file and Path(args.output_file).suffix.lower() != extension:
        parser.error(f"--output_file must have a {extension} extension")
    return args



def setup_logging(path, level):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    # Replace only handlers owned by this entry point.
    for handler in root.handlers[:]:
        if getattr(handler, "mimicmotion_cli", False):
            root.removeHandler(handler)
            handler.close()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(path, mode="a", encoding="utf-8")):
        handler.mimicmotion_cli = True
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(level)


def check_file(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"File not found: {path}")
    with path.open("rb") as stream:
        if not stream.read(1):
            raise ValueError(f"Empty file: {path}")
    return str(path)


def check_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=path):
        pass


def plan_tasks(args, run_id):
    args.ckpt_path = check_file(args.ckpt_path)
    for name in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
        check_file(Path("models/DWPose") / name)
    model = Path(args.base_model_path).expanduser()
    if model.is_dir():
        args.base_model_path = str(model.resolve())
    elif args.base_model_path.startswith(("/", ".", "~")):
        raise ValueError(f"Base model directory not found: {model}")
    extension = ".mov" if args.transparent_background else ".mp4"
    tasks = []
    for index, video in enumerate(args.ref_video_path):
        image = args.ref_image_path[0 if len(args.ref_image_path) == 1 else index]
        output = (Path(args.output_file) if args.output_file else
                  Path(args.output_dir) / f"{Path(video).stem}_{run_id}_{index + 1:03d}{extension}")
        output = output.expanduser().resolve()
        task = dict(video=video, image=image, output=output, error=None)
        try:
            task["video"], task["image"] = check_file(video), check_file(image)
            protected = {Path(task["video"]), Path(task["image"]), Path(args.ckpt_path)}
            if output in protected or output.with_suffix(".json") in protected:
                raise ValueError("Output must not replace an input or checkpoint")
            check_directory(output.parent)
            if not args.overwrite and (output.exists() or output.with_suffix(".json").exists()):
                raise ValueError(f"Output already exists: {output}; use --overwrite to replace it")
        except (ValueError, OSError) as exc:
            task["error"] = str(exc)
            if not args.continue_on_error:
                raise ValueError(f"Task {index + 1}: {exc}") from exc
        tasks.append(task)
    return tasks


def resolve_runtime(args, torch):
    name = args.device
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid --device: {name}") from exc
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA devices are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index out of range: {index}")
        device = torch.device(f"cuda:{index}")
    dtype = args.dtype or ("float16" if device.type == "cuda" else "float32")
    if device.type == "cpu" and dtype == "float16":
        raise ValueError("Use --dtype float32 with CPU")
    args.device, args.dtype = str(device), dtype
    return device, getattr(torch, dtype)


def probe_media(task, args):
    from PIL import Image
    import decord
    with Image.open(task["image"]) as image:
        image.verify()
    reader = decord.VideoReader(task["video"], ctx=decord.cpu(0))
    source_fps = float(reader.get_avg_fps())
    if len(reader) == 0 or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Video has no frames or valid frame rate")
    stride = 1 if args.mode == 'replace' else args.sample_stride * max(1, int(source_fps / 24))
    source_stream = None
    if args.mode == 'replace':
        import numpy as np
        from mimicmotion.utils.replacement import probe_source, mask_paths
        source_stream = probe_source(task['video'])
        timestamps = np.asarray(reader.get_frame_timestamp(list(range(len(reader)))))
        if len(timestamps) > 1 and not np.allclose(np.diff(timestamps[:,0]),1/source_fps,atol=1e-4,rtol=0.01):
            raise ValueError('Variable-frame-rate footage must be normalized to CFR before replacement')
        actual_shape = reader[0].shape
        if tuple(actual_shape[:2]) != (source_stream['height'],source_stream['width']):
            raise ValueError('Decoded dimensions disagree with source metadata')
        for masks in (args.source_masks,args.replacement_masks,args.occlusion_masks):
            if masks:
                mask_paths(masks,len(reader))
    sampled = len(range(0, len(reader), stride))
    if args.transparent_background and args.replacement_masks and args.mode == "generate":
        from mimicmotion.utils.replacement import mask_paths
        mask_paths(args.replacement_masks, sampled)
    if sampled + 1 < args.num_frames:
        raise ValueError(f"Video yields {sampled} sampled frames plus one reference frame; "
                         f"at least {args.num_frames} total required. Reduce --num_frames or --sample_stride.")
    return dict(source_frames=len(reader), source_fps=source_fps,
                effective_sample_stride=stride, sampled_frames=sampled, source_stream=source_stream)


def publish(temp, target, overwrite):
    if overwrite:
        os.replace(temp, target)
    else:
        # A hard link publishes a completed file without a check-then-replace race.
        os.link(temp, target)
        temp.unlink()


def write_json(path, record, overwrite=True):
    fd, name = tempfile.mkstemp(prefix=".mimicmotion-", suffix=".json", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        publish(temp, path, overwrite)
    finally:
        temp.unlink(missing_ok=True)


def save_result(frames, output, args, record):
    from mimicmotion.utils.utils import save_to_mp4
    fd, name = tempfile.mkstemp(prefix=".mimicmotion-", suffix=".mp4", dir=output.parent)
    os.close(fd)
    temp = Path(name)
    try:
        save_to_mp4(frames, str(temp), fps=args.fps)
        if not temp.stat().st_size:
            raise OSError("Video encoder produced an empty file")
        publish(temp, output, args.overwrite)
        # The summary also retains the record if sidecar publication fails.
        write_json(output.with_suffix(".json"), record, args.overwrite)
    finally:
        temp.unlink(missing_ok=True)


def environment_info():
    versions = {}
    for package in ("torch", "torchvision", "diffusers", "transformers", "numpy", "decord", "onnxruntime", "onnxruntime-gpu"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions)


def _validate_run_logging(args, run_id, output_dir):
    check_directory(output_dir)
    log_path = Path(args.log_file).expanduser().resolve() if args.log_file else output_dir / f"{run_id}.log"
    protected = {Path(p).expanduser().resolve() for p in
                 args.ref_video_path + args.ref_image_path + [args.ckpt_path]}
    if args.output_file:
        protected.update((Path(args.output_file).expanduser().resolve(),
                          Path(args.output_file).expanduser().resolve().with_suffix(".json")))
    if log_path in protected:
        raise ValueError("Log path must differ from inputs, checkpoint, and outputs")
    setup_logging(log_path, args.log_level)


def _prepare_tasks(args, run_id):
    if args.mode == 'replace' or args.transparent_background:
        from mimicmotion.utils.replacement import check_dependencies
        check_dependencies(args)
    tasks = plan_tasks(args, run_id)
    # Media decoding is checked before expensive model construction.
    for task in tasks:
        if task["error"]:
            continue
        try:
            task["media"] = probe_media(task, args)
        except Exception as exc:
            task["error"] = f"Media validation: {exc}"
            if not args.continue_on_error:
                raise ValueError(task["error"]) from exc
    return tasks


def _compute_model_info(args):
    checkpoint = Path(args.ckpt_path)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return dict(base_model=args.base_model_path, checkpoint=str(checkpoint),
                checkpoint_bytes=checkpoint.stat().st_size, checkpoint_sha256=digest.hexdigest())


def _load_pipeline_and_processor(args):
    import torch
    device, dtype = resolve_runtime(args, torch)
    from mimicmotion.utils.geglu_patch import patch_geglu_inplace
    patch_geglu_inplace()
    from mimicmotion.utils.loader import create_pipeline
    from mimicmotion.dwpose.dwpose_detector import DWposeDetector
    started = time.perf_counter()
    pipeline = create_pipeline(args, device, dtype=dtype)
    logger.info("Model loaded in %.2fs; device=%s dtype=%s", time.perf_counter() - started, device, dtype)
    processor = DWposeDetector("models/DWPose/yolox_l.onnx", "models/DWPose/dw-ll_ucoco_384.onnx", device=device)
    return pipeline, processor, device, torch


def _process_task(index, task, args, env, model_info, pipeline, processor, device, torch):
    started = time.perf_counter()
    stage = "validation"
    record = dict(task=index + 1, video=str(task["video"]), image=str(task["image"]),
                  output=str(task["output"]), parameters=vars(args).copy(), environment=env, model=model_info,
                  media=task.get("media"), started_at=datetime.now(timezone.utc).isoformat())
    frames = pose_pixels = image_pixels = None
    replacement_work = None
    replacement_geometry = None
    code = 0
    try:
        if task["error"]:
            raise ValueError(task["error"])
        stage = "preprocess"
        with torch.no_grad():
            if args.mode == 'replace':
                from mimicmotion.utils.replacement import prepare
                replacement_work = Path(tempfile.mkdtemp(prefix='.replacement-', dir=task['output'].parent))
                pose_pixels, image_pixels, replacement_geometry = prepare(task, args, processor, replacement_work)
            else:
                pose_pixels, image_pixels = preprocess(task["video"], task["image"], args.resolution,
                                                      args.sample_stride, processor, args.num_frames)
            processor.release_memory()
            record["preprocess_seconds"] = time.perf_counter() - started
            logger.info("Task %d pose extraction: %.2fs", index + 1, record["preprocess_seconds"])
            stage = "inference"
            inference_start = time.perf_counter()
            frames = run_pipeline(pipeline, image_pixels, pose_pixels, device, args)
            record["inference_seconds"] = time.perf_counter() - inference_start
        record.update(status="success", output_frames=int(frames.shape[0]),
                      generation_seconds=time.perf_counter() - started)
        stage = "output"
        if args.mode == 'replace' or args.transparent_background:
            from mimicmotion.utils.replacement import render
            if replacement_work is None:
                replacement_work = Path(tempfile.mkdtemp(prefix='.transparent-', dir=task['output'].parent))
            pipeline.to('cpu')
            if getattr(device, 'type', None) == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            fd, temp_name = tempfile.mkstemp(prefix='.replacement-', suffix=task['output'].suffix, dir=task['output'].parent)
            os.close(fd)
            temp_video = Path(temp_name)
            try:
                if args.transparent_background:
                    from mimicmotion.utils.transparent import render_transparent
                    record['transparency'] = render_transparent(
                        frames, task, args, replacement_work, replacement_geometry, temp_video, processor)
                else:
                    record['replacement'] = render(frames, task, args, replacement_work, replacement_geometry, temp_video)
                publish(temp_video, task['output'], args.overwrite)
                write_json(task['output'].with_suffix('.json'), record, args.overwrite)
            finally:
                temp_video.unlink(missing_ok=True)
        else:
            save_result(frames, task["output"], args, record)
        logger.info("Task %d saved: %s (%.2fs)", index + 1, task["output"], time.perf_counter() - started)
    except Exception as exc:
        oom = torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError)
        code = 4 if oom else (5 if stage == "output" else (2 if stage == "validation" or (stage == "preprocess" and isinstance(exc, ValueError)) else 3))
        record.update(status="failed", stage=stage, error=str(exc), exit_code=code)
        logger.exception("Task %d failed at %s%s", index + 1, stage,
                         "; reduce resolution or decode_chunk_size" if oom else "")
    finally:
        record["total_seconds"] = time.perf_counter() - started
        if replacement_work is not None:
            if args.keep_intermediates:
                record['intermediates'] = str(replacement_work)
                logger.info('Review intermediates: %s', replacement_work)
            else:
                import shutil
                shutil.rmtree(replacement_work)
        frames = pose_pixels = image_pixels = None
        if processor is not None:
            processor.release_memory()
        if pipeline is not None:
            pipeline.to("cpu")
        if torch is not None and args.device.startswith("cuda"):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    return record, code


def main(args):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_dir = Path(args.output_dir).expanduser().resolve()
    _validate_run_logging(args, run_id, output_dir)
    tasks = _prepare_tasks(args, run_id)

    pipeline = processor = torch = device = None
    records = []
    status = 0
    summary = output_dir / f"{run_id}_summary.json"
    env = environment_info()
    model_info = _compute_model_info(args)

    try:
        if any(not task["error"] for task in tasks):
            pipeline, processor, device, torch = _load_pipeline_and_processor(args)

        for index, task in enumerate(tasks):
            record, code = _process_task(index, task, args, env, model_info, pipeline, processor, device, torch)
            records.append(record)
            if code != 0:
                status = status or code
                if not args.continue_on_error:
                    break
    except KeyboardInterrupt:
        status = 130
        raise
    except Exception as exc:
        status = status or (4 if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError)
                            else (2 if isinstance(exc, ValueError) else 3))
        records.append(dict(status="failed", stage="initialization_or_cleanup", error=str(exc), exit_code=status))
        raise
    finally:
        write_json(summary, dict(run_id=run_id, results=records,
                                unprocessed=len(tasks) - sum("task" in record for record in records), exit_code=status))
        logger.info("Run summary: %s", summary)
    return status


def cli(argv=None):
    args = parse_args(argv)
    try:
        return main(args)
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except ValueError:
        logger.exception("Invalid input or runtime option")
        return 2
    except OSError:
        logger.exception("File or output operation failed")
        return 5
    except Exception as exc:
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
            logger.exception("GPU out of memory")
            return 4
        logger.exception("Model initialization or runtime failed")
        return 3


if __name__ == "__main__":
    sys.exit(cli())
