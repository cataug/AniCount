#!/usr/bin/env python3
"""AniCount: tiled inference with local DMD, OWL-C and OWL-D checkpoints.

Controller/setup use only the standard library. ML imports run in isolated
worker processes. Default project: /home/tahiti/AniCount. See README_RU.md.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import types
import venv


VERSION = "1.0"
HERE = Path(__file__).resolve().parent
VENDOR = HERE / "anicount_vendor"
LOCKFILE = HERE / "requirements_inference.lock"
MODEL_INFO = {
    "dmd": {
        "file": "DroneMegaDetector_RFDETRMedium_2026.10.04.pth",
        "type": "bbox", "title": "DroneMegaDetector",
        "checksum": "sha256:a50d654c2c5b2739bf0874450a834bcc2a6190eb72eba31f926294e3d2f44c01",
        "checksum_source": "SHA256 of official v1.1 GitHub asset downloaded and tested for this package",
    },
    "owl-c": {
        "file": "OWL-C.pth", "type": "point", "title": "OWL-C",
        "checksum": "md5:4d1cb40b4ee16d531ded09c16333d73b",
        "checksum_source": "Zenodo 20802844 published checksum",
    },
    "owl-d": {
        "file": "OWL-D.pth", "type": "point", "title": "OWL-D",
        "checksum": "md5:30ff57bba505de4be743c2e0ef0c23ed",
        "checksum_source": "Zenodo 20802844 published checksum",
    },
}
PINS = [
    "rfdetr==1.11.1", "transformers==5.1.0", "supervision==0.30.6",
    "numpy==1.26.4", "opencv-python-headless==4.10.0.84", "Pillow==11.3.0",
]
DETECTION_FIELDS = ["frame", "time_s", "kind", "detection_index", "score",
                    "x", "y", "x1", "y1", "x2", "y2", "category"]
COUNT_FIELDS = ["frame", "time_s", "count", "inference_s", "tiles"]


def say(message):
    print(message, flush=True)


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def timestamp():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def parse_time(value):
    try:
        parts = [float(p) for p in str(value).split(":")]
        if not 1 <= len(parts) <= 3:
            raise ValueError()
        seconds = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError()
        return seconds
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Time must be nonnegative seconds or MM:SS[.sss]") from exc


def source_path(root, requested):
    if requested:
        path = Path(requested).expanduser()
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise RuntimeError(f"Video not found: {path}")
        return path.resolve()
    exact = root / "NG South Sudan 1920x1080 (1).m4v.mp4"
    if exact.is_file():
        return exact
    candidates = [p for p in root.iterdir() if p.is_file() and
                  p.name.lower().startswith("ng south sudan") and
                  p.suffix.lower() in {".mp4", ".m4v", ".mov", ".mkv", ".avi"}]
    if len(candidates) == 1:
        return candidates[0].resolve()
    inventory = "\n".join(str(p.name) for p in candidates) or "none"
    raise RuntimeError(f"Cannot select the HD source automatically. Pass --video 'filename'. Candidates:\n{inventory}")


def gpu_inventory():
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode:
            return []
        return [{"index": int(parts[0]), "name": parts[1], "free_gib": float(parts[2]) / 1024}
                for row in result.stdout.splitlines()
                if len(parts := [p.strip() for p in row.split(",")]) == 3]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []


def select_device(requested, min_free_gib):
    cards = gpu_inventory()
    for card in cards:
        say(f"[GPU] {card['index']}: {card['name']} | free {card['free_gib']:.2f} GiB")
    if requested == "cpu":
        return "cpu", "explicit CPU selection"
    if requested == "auto":
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            return "cpu", "CUDA_VISIBLE_DEVICES is empty"
        # On the user's single-A100 server this maps to cuda:0. Respect an
        # inherited list of physical GPU IDs rather than selecting a hidden GPU.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        allowed = visible.split(",") if visible else [str(c["index"]) for c in cards]
        for logical, physical in enumerate(allowed):
            for card in cards:
                if str(card["index"]) == physical.strip() and card["free_gib"] >= min_free_gib:
                    index = logical if visible else card["index"]
                    return f"cuda:{index}", f"at least {min_free_gib:g} GiB free before loading"
        return "cpu", f"no visible GPU with {min_free_gib:g} GiB free; CPU inference"
    if requested == "cuda" or (requested.startswith("cuda:") and requested[5:].isdigit()):
        return requested, "explicit CUDA selection"
    raise RuntimeError("--device must be auto, cpu, cuda, or cuda:N")


def check_environment(python):
    check = (
        "import cv2,torch,torchvision,rfdetr,transformers,supervision;"
        "from importlib.metadata import version;"
        "print('torch='+torch.__version__+'; rfdetr='+version('rfdetr')+'; transformers='+transformers.__version__);"
        "assert torch.__version__.split('+')[0]=='2.6.0';"
        "assert version('rfdetr')=='1.11.1';"
        "assert transformers.__version__=='5.1.0'"
    )
    result = subprocess.run([str(python), "-c", check], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Inference environment check failed:\n" + result.stdout + result.stderr)
    say("[ENV] " + result.stdout.strip())


def ensure_environment(root, requested_python=None, build="auto"):
    if requested_python:
        # Keep the venv symlink path: resolving bin/python to the system binary
        # would discard that interpreter's venv site-packages.
        python = Path(requested_python).expanduser().absolute()
        check_environment(python)
        return python
    env = root / ".venv_anicount"
    python = env / "bin/python"
    marker = env / "anicount_ready.json"
    if not LOCKFILE.is_file():
        raise RuntimeError(f"Extract the entire AniCount package: missing {LOCKFILE}")
    signature = {"torch": "2.6.0", "torchvision": "0.21.0", "pins": PINS, "version": VERSION,
                 "dependency_lock_sha256": hashlib.sha256(LOCKFILE.read_bytes()).hexdigest()}
    if marker.is_file() and json.loads(marker.read_text()).get("requirements") == signature:
        if not python.is_file():
            raise RuntimeError(f"Environment marker exists but interpreter is missing: {python}")
        return python
    if not (3, 11) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("Run setup with Python 3.11 or 3.12; the server's python3.12 is supported")
    if env.exists() and not python.is_file():
        raise RuntimeError(f"Existing directory is not a usable venv: {env}. Choose a different --root or inspect it.")
    if not python.is_file():
        say(f"[SETUP] Creating project environment: {env}")
        venv.EnvBuilder(with_pip=True).create(env)
    chosen = ("gpu" if gpu_inventory() else "cpu") if build == "auto" else build
    index = "https://download.pytorch.org/whl/" + ("cu124" if chosen == "gpu" else "cpu")
    constraints = env / "anicount_constraints.txt"
    constraints.write_text("torch==2.6.0\ntorchvision==0.21.0\n" + "\n".join(PINS) + "\n")
    commands = [
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        [str(python), "-m", "pip", "install", "torch==2.6.0", "torchvision==0.21.0", "--index-url", index],
        [str(python), "-m", "pip", "install", "-c", str(constraints), "-r", str(LOCKFILE)],
    ]
    for command in commands:
        say("[SETUP] " + " ".join(command))
        subprocess.run(command, check=True)
    check_environment(python)
    write_json(marker, {"requirements": signature, "build": chosen, "python": str(python)})
    say(f"[READY] {python}")
    return python


def verify_weights(path, name):
    info = MODEL_INFO[name]
    say(f"[VERIFY] {path.name}: checking {info['checksum'].split(':')[0]}")
    algorithm, expected = info["checksum"].split(":", 1)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, algorithm).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Checkpoint checksum mismatch: {path}. Existing file retained.")
    return {**info, "path": str(path), "bytes": path.stat().st_size, "verified_checksum": f"{algorithm}:{actual}"}


def verify_vendor():
    record_path = VENDOR / "SOURCE.json"
    if not record_path.is_file():
        raise RuntimeError(f"Extract the entire AniCount package: missing {record_path}")
    record = json.loads(record_path.read_text())
    for relative, expected in record["files"].items():
        path = VENDOR / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Missing or modified bundled architecture file: {path}")
    return record["commit"]


def owl_modules(name):
    # Upstream animaloc.__init__ eagerly imports its training/geospatial stack.
    # Load its unmodified inference modules in a private namespace instead.
    namespace = "_anicount_owl"
    for suffix, directory in [("", "animaloc"), (".models", "animaloc/models"),
                              (".utils", "animaloc/utils"), (".eval", "animaloc/eval")]:
        key = namespace + suffix
        if key not in sys.modules:
            package = types.ModuleType(key)
            package.__path__ = [str(VENDOR / directory)]
            package.__package__ = key
            sys.modules[key] = package
    if name == "owl-d":
        sys.path.insert(0, str(VENDOR / "dinov3"))
    model_module = importlib.import_module(namespace + (".models.owl_c" if name == "owl-c" else ".models.owl_d"))
    lmds = importlib.import_module(namespace + ".eval.lmds").HerdNet_Detection_Branch_LMDS
    return (model_module.OWLC if name == "owl-c" else model_module.OWLD_H), lmds


def tile_starts(length, size, overlap):
    if length <= size:
        return [0]
    result = list(range(0, length - size + 1, size - overlap))
    if result[-1] != length - size:
        result.append(length - size)
    return result


def tiles_for(width, height, size, overlap):
    return [(x, y) for y in tile_starts(height, size, overlap)
            for x in tile_starts(width, size, overlap)]


def box_nms(boxes, threshold):
    """Class-filtered bbox IoU NMS; works without compiled torchvision ops."""
    import numpy as np
    if not boxes:
        return []
    a = np.asarray(boxes, dtype=np.float32)
    order = np.argsort(-a[:, 4], kind="stable")
    kept = []
    while len(order):
        first = int(order[0])
        kept.append(first)
        if len(order) == 1:
            break
        rest = order[1:]
        lo = np.maximum(a[first, :2], a[rest, :2])
        hi = np.minimum(a[first, 2:4], a[rest, 2:4])
        intersection = np.maximum(hi - lo, 0).prod(axis=1)
        area_first = np.maximum(a[first, 2:4] - a[first, :2], 0).prod()
        area_rest = np.maximum(a[rest, 2:4] - a[rest, :2], 0).prod(axis=1)
        iou = intersection / np.maximum(area_first + area_rest - intersection, 1e-12)
        order = rest[iou <= threshold]
    return [boxes[i] for i in kept]


class Detector:
    def __init__(self, name, checkpoint, device, config):
        import torch
        self.name, self.device, self.config = name, device, config
        if name == "dmd":
            import rfdetr
            self.model = rfdetr.from_checkpoint(
                str(checkpoint), device=device, resolution=config["dmd_size"], trust_checkpoint=True,
            )
            names = self.model.class_names
            if isinstance(names, dict):
                animal = [int(k) for k, v in names.items() if str(v).lower() in {"animal", "animals"}]
            else:
                animal = [i for i, value in enumerate(names) if str(value).lower() in {"animal", "animals"}]
            if animal != [1]:
                raise RuntimeError(f"Unexpected DMD class mapping: {names}; expected animal=1")
            say(f"[MODEL] DMD | true boxes | animal class=1 | inference size={config['dmd_size']}")
        else:
            constructor, peak_extractor = owl_modules(name)
            kwargs = {"pretrained": False, "head_conv": 64, "down_ratio": 2}
            if name == "owl-d":
                kwargs.update(readout_type="ignore", freeze_backbone=True,
                              dinov3_root=str(VENDOR / "dinov3"))
            say(f"[MODEL] Constructing {name}; pretrained=False (backbone is in the local checkpoint)")
            self.model = constructor(**kwargs)
            checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
            state = checkpoint_data.get("model_state_dict")
            if not isinstance(state, dict) or not state:
                raise RuntimeError("OWL checkpoint must contain a nonempty model_state_dict")
            if all(key.startswith("model.") for key in state):
                state = {key[6:]: value for key, value in state.items()}
            # Inference has no optimizer: assigning mmap-backed tensors avoids
            # keeping a second full CPU copy of the 3.5 GB OWL-D checkpoint.
            self.model.load_state_dict(state, strict=True, assign=True)
            del checkpoint_data, state
            self.model.eval().requires_grad_(False).to(device)
            self.peaks = peak_extractor(up=False, kernel_size=(3, 3),
                                        adapt_ts=config["owl_adapt"], neg_ts=config["owl_negative"])
            say(f"[MODEL] {name}: strict checkpoint load OK | localization points")

    def predict(self, frame):
        import cv2
        import numpy as np
        import torch
        config = self.config
        height, width = frame.shape[:2]
        tile = config["tile"]
        # Make all OWL tile offsets even and cover small/odd-sized images.
        padded_width = max(tile, math.ceil(width / 32) * 32)
        padded_height = max(tile, math.ceil(height / 32) * 32)
        image = cv2.copyMakeBorder(frame, 0, padded_height - height, 0, padded_width - width,
                                   cv2.BORDER_REPLICATE)
        positions = tiles_for(padded_width, padded_height, tile, config["overlap"])
        boxes = []
        heat_sum = heat_count = None
        if self.name != "dmd":
            heat_sum = np.zeros((padded_height // 2, padded_width // 2), dtype=np.float32)
            heat_count = np.zeros_like(heat_sum)
        last_log = time.monotonic()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        begin = time.perf_counter()
        with torch.inference_mode():
            for index, (x, y) in enumerate(positions, 1):
                rgb = cv2.cvtColor(image[y:y + tile, x:x + tile], cv2.COLOR_BGR2RGB)
                if self.name == "dmd":
                    from PIL import Image
                    prediction = self.model.predict(Image.fromarray(rgb), threshold=config["dmd_conf"],
                                                    include_source_image=False)
                    for box, score, category in zip(prediction.xyxy, prediction.confidence, prediction.class_id):
                        if int(category) != 1 or not np.isfinite(box).all() or not math.isfinite(float(score)):
                            continue
                        x1, y1, x2, y2 = (float(v) for v in box)
                        x1, x2 = max(0., min(float(width), x1 + x)), max(0., min(float(width), x2 + x))
                        y1, y2 = max(0., min(float(height), y1 + y)), max(0., min(float(height), y2 + y))
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2, float(score)])
                else:
                    a = rgb.astype(np.float32) / 255.
                    a = (a - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
                    tensor = torch.from_numpy(a).permute(2, 0, 1).contiguous().unsqueeze(0).to(self.device)
                    output = self.model(tensor)
                    # OWL-C returns a heatmap; OWL-D returns (heatmap, features).
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    if tuple(output.shape) != (1, 1, tile // 2, tile // 2) or not torch.isfinite(output).all():
                        raise RuntimeError(f"Unexpected/non-finite OWL heatmap: {tuple(output.shape)}")
                    patch_map = output[0, 0].float().cpu().numpy()
                    yy, xx = y // 2, x // 2
                    heat_sum[yy:yy + tile // 2, xx:xx + tile // 2] += patch_map
                    heat_count[yy:yy + tile // 2, xx:xx + tile // 2] += 1.
                    del tensor, output
                if index == len(positions) or time.monotonic() - last_log >= 10:
                    say(f"[TILES] {self.name}: {index}/{len(positions)}")
                    last_log = time.monotonic()
            detections = []
            if self.name == "dmd":
                for x1, y1, x2, y2, score in box_nms(boxes, config["nms_iou"]):
                    detections.append({"kind": "bbox", "score": score,
                                       "x": (x1+x2)/2, "y": (y1+y2)/2,
                                       "x1": x1, "y1": y1, "x2": x2, "y2": y2, "category": "animal"})
                heatmap = None
            else:
                if np.any(heat_count == 0):
                    raise RuntimeError("Tiling left uncovered pixels in the stitched heatmap")
                heatmap = (heat_sum / heat_count)[:math.ceil(height/2), :math.ceil(width/2)]
                heatmap = np.ascontiguousarray(heatmap)
                _, locations, _, scores = self.peaks(torch.from_numpy(heatmap)[None, None])
                for (yy, xx), score in zip(locations[0], scores[0]):
                    xx, yy = float(xx * 2), float(yy * 2)
                    if score >= config["owl_conf"] and 0 <= xx < width and 0 <= yy < height:
                        detections.append({"kind": "point", "score": float(score),
                                           "x": xx, "y": yy, "category": "animal"})
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        detections.sort(key=lambda d: (d["y"], d["x"]))
        return detections, heatmap, elapsed, len(positions)


def draw_frame(frame, detections, name, time_s, config):
    import cv2
    import numpy as np
    height, width = frame.shape[:2]
    result = frame.copy()
    for d in detections:
        point = (int(round(d["x"])), int(round(d["y"])))
        if d["kind"] == "bbox":
            cv2.rectangle(result, (max(0, int(d["x1"])), max(0, int(d["y1"]))),
                          (min(width-1, int(math.ceil(d["x2"]))), min(height-1, int(math.ceil(d["y2"])))),
                          (40, 255, 60), 1, cv2.LINE_AA)
        else:
            cv2.circle(result, point, 3, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(result, point, 2, (40, 255, 255), -1, cv2.LINE_AA)
    footer = np.full((48, width, 3), 20, dtype=np.uint8)
    threshold = config["dmd_conf"] if name == "dmd" else config["owl_conf"]
    text = (f"{MODEL_INFO[name]['title']} | {MODEL_INFO[name]['type']} | "
            f"count={len(detections)} | source {time_s:.3f}s | threshold={threshold:g}")
    cv2.putText(footer, text, (12, 31), cv2.FONT_HERSHEY_SIMPLEX,
                min(.7, width/1300), (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([result, footer])


class VideoSink:
    def __init__(self, path, width, height, fps):
        import cv2
        self.path, self.pipe, self.writer, self.log = path, None, None, None
        if shutil.which("ffmpeg"):
            codecs = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
            if "libx264" in codecs.stdout:
                self.log = path.with_suffix(".ffmpeg.log").open("wb")
                self.pipe = subprocess.Popen([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                    "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.9f}",
                    "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2",
                    "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(path),
                ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        if self.pipe is None:
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError("Cannot create output MP4. Install ffmpeg with libx264 support.")
            say("[VIDEO] ffmpeg/libx264 unavailable; using OpenCV MPEG-4 codec")

    def write(self, frame):
        if self.pipe:
            try:
                self.pipe.stdin.write(frame.tobytes())
            except BrokenPipeError as exc:
                raise RuntimeError(f"Video encoder failed; see {self.path.with_suffix('.ffmpeg.log')}") from exc
        else:
            self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.pipe is not None:
            pipe, self.pipe = self.pipe, None
            try:
                if pipe.stdin:
                    pipe.stdin.close()
                code = pipe.wait(timeout=60)
                if code:
                    raise RuntimeError(f"ffmpeg exited with {code}; see {self.path.with_suffix('.ffmpeg.log')}")
            finally:
                if self.log:
                    self.log.close()


def save_image(path, image):
    import cv2
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def run_worker(config_path, name, force_device=None):
    config = json.loads(Path(config_path).read_text())
    output = Path(config["out"]) / name
    output.mkdir(parents=True, exist_ok=True)
    device, reason = select_device(force_device or config["device"], config["min_free_gb"])
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(config["threads"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MPLBACKEND"] = "Agg"
    metadata = {"model": name, "status": "starting", "device": device, "device_reason": reason,
                "config_signature": config["signature"], "source": config["video"],
                "settings": config, "count_meaning": "detections per sampled frame; not unique animals across video"}
    write_json(output / "metadata.json", metadata)
    say(f"[DEVICE] {name}: {device} | {reason}")
    capture = sink = None
    completed, inference_seconds = [], 0.
    try:
        import cv2
        import numpy as np
        import torch
        torch.set_num_threads(config["threads"])
        torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        cv2.setNumThreads(1)
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("Selected CUDA but this environment/driver has no usable CUDA device. Try --device cpu.")
            free, _ = torch.cuda.mem_get_info(torch.device(device))
            if free / 2**30 < config["min_free_gb"]:
                raise torch.cuda.OutOfMemoryError(f"GPU now has only {free/2**30:.2f} GiB free")
            torch.cuda.reset_peak_memory_stats(device)
        capture = cv2.VideoCapture(config["video"])
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV cannot open the video: {config['video']}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or total <= 0:
            raise RuntimeError("Video has invalid FPS/frame count; remux it with ffmpeg before processing")
        duration = total / fps
        if config["at"] is not None:
            first = int(round(config["at"] * fps))
            final = first + 1
        else:
            first = int(math.ceil(config["start"] * fps - 1e-7))
            final = min(total, int(math.ceil(config["end"] * fps - 1e-7)))
        if first < 0 or first >= total or final <= first:
            raise RuntimeError(f"Requested interval is outside the video ({duration:.3f}s, {total} frames)")
        expected = math.ceil((final - first) / config["stride"])
        if config["max_frames"]:
            expected = min(expected, config["max_frames"])
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
            raise RuntimeError(f"Video backend cannot seek to frame {first}")
        say(f"[VIDEO] {config['video']} | {fps:.6f} FPS | {duration:.3f}s | {expected} selected frames")
        checkpoint = Path(config["weights_dir"]) / MODEL_INFO[name]["file"]
        checkpoint_info = verify_weights(checkpoint, name)
        metadata.update(status="loading", checkpoint=checkpoint_info, torch=torch.__version__,
                        fps=fps, video_duration_s=duration, requested_frame_count=expected)
        write_json(output / "metadata.json", metadata)
        detector = Detector(name, checkpoint, device, config)
        (output / "frames").mkdir(exist_ok=True)
        references = Path(config["out"]) / "reference_frames"
        references.mkdir(exist_ok=True)
        preview_indices = {int(round(value*fps)) for value in config["preview_times"]}
        start_clock = time.monotonic()
        with (output / "detections.csv").open("w", newline="", encoding="utf-8") as det_stream, \
             (output / "counts.csv").open("w", newline="", encoding="utf-8") as count_stream:
            det_writer = csv.DictWriter(det_stream, fieldnames=DETECTION_FIELDS)
            count_writer = csv.DictWriter(count_stream, fieldnames=COUNT_FIELDS)
            det_writer.writeheader()
            count_writer.writeheader()
            frame_index = first
            processed = 0
            while frame_index < final and processed < expected:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Video decoding failed at source frame {frame_index}; partial outputs retained")
                source_time = frame_index / fps
                say(f"[FRAME] {name} {processed+1}/{expected} | frame={frame_index} | source={source_time:.3f}s")
                detections, heatmap, seconds, tile_count = detector.predict(frame)
                annotated = draw_frame(frame, detections, name, source_time, config)
                for index, detection in enumerate(detections, 1):
                    row = {"frame": frame_index, "time_s": round(source_time, 6), "detection_index": index,
                           **{k: round(v, 6) if isinstance(v, float) else v for k, v in detection.items()}}
                    det_writer.writerow(row)
                count_writer.writerow({"frame": frame_index, "time_s": round(source_time, 6),
                                       "count": len(detections), "inference_s": round(seconds, 5), "tiles": tile_count})
                det_stream.flush()
                count_stream.flush()
                completed.append(len(detections))
                inference_seconds += seconds
                preview = (processed == 0 or processed == expected-1 or frame_index in preview_indices
                           or processed % max(1, int(round(fps/config["stride"]))) == 0)
                if preview or config["save_all_frames"] or config["at"] is not None:
                    stem = f"frame_{frame_index:07d}_{source_time:010.3f}s"
                    save_image(output / "frames" / (stem + ".png"), annotated)
                    reference = references / (stem + ".png")
                    if not reference.exists():
                        save_image(reference, frame)
                    if heatmap is not None:
                        np.save(output / "frames" / (stem + "_heatmap.npy"), heatmap)
                if config["at"] is None and not config["no_video"]:
                    if sink is None:
                        height, width = annotated.shape[:2]
                        sink = VideoSink(output / "annotated.mp4", width, height, fps/config["stride"])
                    sink.write(annotated)
                processed += 1
                metadata.update(status="running", frames_processed=processed, last_frame=frame_index,
                                last_count=len(detections), inference_seconds=round(inference_seconds, 4))
                write_json(output / "metadata.json", metadata)
                rate = processed / max(time.monotonic()-start_clock, 1e-9)
                eta = (expected-processed) / max(rate, 1e-9)
                say(f"[RESULT] {name}: count={len(detections)} | {seconds:.2f}s/frame | {rate:.3f} processing FPS | ETA {eta/60:.1f} min")
                if processed >= expected:
                    break
                for _ in range(config["stride"]-1):
                    if frame_index + 1 >= final:
                        break
                    if not capture.grab():
                        raise RuntimeError("Video decoding failed while skipping frames")
                    frame_index += 1
                frame_index += 1
        if sink is not None:
            sink.close()
            sink = None
        metadata.update(status="complete", frames_processed=len(completed),
                        count_min=min(completed), count_max=max(completed),
                        count_median=float(np.median(completed)),
                        inference_seconds=round(inference_seconds, 4),
                        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.startswith("cuda") else None,
                        interpretation="Unannotated inference. Counts are model predictions; no accuracy/recall claim. OWL scores are not calibrated probabilities.")
        write_json(output / "metadata.json", metadata)
        say(f"[DONE] {name}: {len(completed)} frames | count range {min(completed)}..{max(completed)} | {output}")
        return 0
    except BaseException as error:
        is_oom = "torch" in sys.modules and isinstance(error, sys.modules["torch"].cuda.OutOfMemoryError)
        metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        frames_processed=len(completed), error=str(error),
                        traceback=traceback.format_exc(), out_of_memory=is_oom)
        write_json(output / "metadata.json", metadata)
        traceback.print_exc()
        return 130 if isinstance(error, KeyboardInterrupt) else (3 if is_oom else 1)
    finally:
        if capture is not None:
            capture.release()
        if sink is not None:
            try:
                sink.close()
            except Exception as error:
                say(f"[ENCODER] {error}; CSV and preview frames retained")


def child_process(command, logfile, environment):
    with logfile.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=environment)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            return process.wait()
        except KeyboardInterrupt:
            # Only the worker created by this controller is signalled.
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/home/tahiti/AniCount"))
    p.add_argument("--setup", action="store_true", help="Install/check the project environment and stop")
    p.add_argument("--install-build", choices=["auto", "gpu", "cpu"], default="auto")
    p.add_argument("--python", type=Path, help="Use an already compatible interpreter without installing into it")
    p.add_argument("--video", help="HD video path; default is the known NG South Sudan filename")
    p.add_argument("--weights-dir", type=Path)
    p.add_argument("--models", nargs="+", choices=list(MODEL_INFO), default=list(MODEL_INFO))
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--min-free-gb", type=float, default=8.)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--start", type=parse_time, default=61.)
    p.add_argument("--end", type=parse_time, default=68.)
    p.add_argument("--at", type=parse_time, help="Process exactly one full source frame, e.g. 64.5")
    p.add_argument("--stride", type=int, default=1, help="Process every Nth source frame; output duration is preserved")
    p.add_argument("--max-frames", type=int, default=0, help="Optional test limit; 0 means the full interval")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--overlap", type=int, default=160)
    p.add_argument("--dmd-size", type=int, default=640, help="RF-DETR input resolution; default 640 for a practical first run, 1280 is checkpoint-native")
    p.add_argument("--dmd-conf", type=float, default=.15)
    p.add_argument("--owl-conf", type=float, default=.20)
    p.add_argument("--owl-adapt", type=float, default=.30)
    p.add_argument("--owl-negative", type=float, default=.10)
    p.add_argument("--nms-iou", type=float, default=.45)
    p.add_argument("--preview-times", type=parse_time, nargs="+", default=[64.5])
    p.add_argument("--save-all-frames", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--out", type=Path)
    p.add_argument("--resume", action="store_true", help="Reuse --out and skip completed models with identical settings")
    p.add_argument("--_worker-config", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_worker-model", choices=list(MODEL_INFO), help=argparse.SUPPRESS)
    p.add_argument("--_worker-device", help=argparse.SUPPRESS)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args._worker_config:
        return run_worker(args._worker_config, args._worker_model, args._worker_device)
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".anicount_inference.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another AniCount setup/inference is active in this project directory") from error
        if args.setup:
            ensure_environment(root, args.python, args.install_build)
            verify_vendor()
            return 0
        if args.threads < 1 or args.stride < 1 or args.max_frames < 0:
            p.error("--threads and --stride must be positive; --max-frames must be >= 0")
        if args.tile < 64 or args.tile % 32 or args.overlap < 0 or args.overlap >= args.tile or args.overlap % 2:
            p.error("--tile must be a multiple of 32 >= 64; --overlap must be even and 0 <= overlap < tile")
        if args.dmd_size < 64 or args.dmd_size % 64:
            p.error("--dmd-size must be a positive multiple of 64 >= 64")
        if args.at is None and args.end <= args.start:
            p.error("--end must be greater than --start")
        for field in ("dmd_conf", "owl_conf", "owl_adapt", "owl_negative", "nms_iou"):
            value = getattr(args, field)
            if not math.isfinite(value) or not 0 < value <= 1:
                p.error(f"--{field.replace('_','-')} must be in (0,1]")
        if not math.isfinite(args.min_free_gb) or args.min_free_gb <= 0:
            p.error("--min-free-gb must be positive")
        if args.resume and args.out is None:
            p.error("--resume requires the exact previous --out directory")
        video = source_path(root, args.video)
        weights = (args.weights_dir or root / "weights").expanduser().resolve()
        for name in args.models:
            checkpoint = weights / MODEL_INFO[name]["file"]
            if not checkpoint.is_file():
                raise RuntimeError(f"Missing local checkpoint: {checkpoint}. This inference script does not download models.")
        vendor_commit = verify_vendor()
        python = ensure_environment(root, args.python, args.install_build)
        output = (args.out or root / "outputs" / (("probe_" if args.at is not None else "video_") + timestamp())).expanduser().resolve()
        config = {k: v for k, v in vars(args).items() if not k.startswith("_") and k not in
                  {"python", "setup", "install_build", "resume", "root", "out", "weights_dir", "models"}}
        config.update(root=str(root), video=str(video), out=str(output), weights_dir=str(weights),
                      source_size=video.stat().st_size, source_mtime_ns=video.stat().st_mtime_ns,
                      vendor_commit=vendor_commit, program_version=VERSION,
                      program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        signature_data = {k: v for k, v in config.items() if k != "out"}
        config["signature"] = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise RuntimeError(f"Output directory is not empty: {output}. Use a new --out or --resume.")
        output.mkdir(parents=True, exist_ok=True)
        config_path = output / "run_config.json"
        if args.resume and config_path.is_file():
            previous = json.loads(config_path.read_text())
            if previous.get("signature") != config["signature"]:
                raise RuntimeError("Resume settings/source/code differ from the previous run; use a new --out directory")
        write_json(config_path, config)
        say(f"[OUTPUT] {output}\n[MODELS] {', '.join(args.models)}\n[SOURCE] {video}")
        say("[COUNTS] Per-frame predictions. Counts across video frames are not added to estimate unique animals.")
        results = []
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        for name in args.models:
            folder = output / name
            if folder.exists() and any(folder.iterdir()):
                old_meta = folder / "metadata.json"
                meta = json.loads(old_meta.read_text()) if old_meta.is_file() else {}
                if args.resume and meta.get("status") == "complete" and meta.get("config_signature") == config["signature"]:
                    say(f"[SKIP] Completed {name}: {folder}")
                    results.append({"model": name, "status": "complete", "path": str(folder), "reused": True})
                    continue
                backup = output / (name + ".previous_" + timestamp())
                folder.rename(backup)
                say(f"[RETAINED] Previous partial output: {backup}")
            folder.mkdir(parents=True, exist_ok=True)
            command = [str(python), "-u", str(Path(__file__).resolve()),
                       "--_worker-config", str(config_path), "--_worker-model", name]
            code = child_process(command, folder / "run.log", environment)
            meta_path = folder / "metadata.json"
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            # A CPU retry is explicit in the log and remains the same model.
            if args.device == "auto" and meta.get("device", "").startswith("cuda") and (code == 3 or code < 0):
                backup = output / (name + ".gpu_failed_" + timestamp())
                folder.rename(backup)
                folder.mkdir()
                say(f"[CPU RETRY] {name}: GPU worker failed ({code}); outputs retained in {backup}")
                code = child_process(command + ["--_worker-device", "cpu"], folder / "run.log", environment)
                meta = json.loads((folder / "metadata.json").read_text())
            status = "complete" if code == 0 and meta.get("status") == "complete" else "failed"
            results.append({"model": name, "status": status, "exit_code": code, "path": str(folder),
                            "frames": meta.get("frames_processed", 0), "count_median": meta.get("count_median"),
                            "device": meta.get("device"), "error": meta.get("error")})
            write_json(output / "summary.json", {"results": results, "config": config,
                                                "interpretation": "Per-frame predicted counts, not accuracy or unique population size"})
        write_json(output / "summary.json", {"results": results, "config": config,
                                            "interpretation": "Per-frame predicted counts, not accuracy or unique population size"})
        say("\n[FINISHED] " + " | ".join(f"{r['model']}: {r['status']}" for r in results))
        say(f"[RESULTS] {output}")
        return 0 if all(r["status"] == "complete" for r in results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        say("\n[STOPPED] Partial outputs retained. Resume using the same settings and --out PATH --resume.")
        raise SystemExit(130)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        say(f"[ERROR] {error}")
        raise SystemExit(1)
