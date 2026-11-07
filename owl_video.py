#!/usr/bin/env python3
"""Make an OWL-D video with the same rings as highlight_owl.py. No installers."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import datetime as dt
from fractions import Fraction
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
BASE = dict(tile=512, overlap=160, owl_conf=.2, owl_adapt=.3, owl_negative=.1,
            dmd_size=640, dmd_conf=.15, nms_iou=.45)
DET_FIELDS = ["frame", "time_s", "kind", "detection_index", "score", "x", "y", "category"]


def say(value):
    print(value, flush=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_module(path, name):
    if not path.is_file():
        raise RuntimeError(f"Required existing file is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def video_info(path):
    if shutil.which("ffprobe"):
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
            "-of", "json", str(path)], capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        rate = stream.get("avg_frame_rate", "0/0")
        if rate in {"0/0", "0", "N/A"}:
            rate = stream["r_frame_rate"]
        fps = Fraction(rate)
        duration = float(stream.get("duration") or data["format"]["duration"])
        count = stream.get("nb_frames", "N/A")
        count = int(count) if count not in {"N/A", "0", None} else int(round(duration * float(fps)))
        return int(stream["width"]), int(stream["height"]), fps, count
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source video: {path}")
        return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                Fraction(float(cap.get(cv2.CAP_PROP_FPS))).limit_denominator(100000),
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()


def cache_data(path, wanted, width, height):
    """Reuse only fully committed frames, including genuine empty frames."""
    counts_path = path.parent / "counts.csv"
    if not counts_path.is_file():
        return {}
    expected = {}
    with counts_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                index, count = int(row["frame"]), int(row["count"])
                if index in wanted and count >= 0:
                    expected[index] = count
            except (TypeError, ValueError, KeyError):
                continue
    points = defaultdict(list)
    invalid = set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                index = int(row["frame"])
                if index not in expected:
                    continue
                x, y = float(row["x"]), float(row["y"])
                score = float(row.get("score") or row.get("dscores") or 0.)
                if row.get("kind", "point") != "point" or not (math.isfinite(x+y+score)
                        and 0 <= x < width and 0 <= y < height):
                    invalid.add(index)
                    continue
                points[index].append(dict(kind="point", score=score, x=x, y=y, category="animal"))
            except (TypeError, ValueError, KeyError):
                if "index" in locals():
                    invalid.add(index)
    return {i: points[i] for i, n in expected.items() if i not in invalid and len(points[i]) == n}


def choose_cache(root, explicit, video, wanted, width, height):
    if explicit:
        directory = Path(explicit).expanduser()
        directory = directory if directory.is_absolute() else root / directory
        if directory.name != "owl-d":
            directory /= "owl-d"
        candidates = [directory / "detections.csv"]
        if not candidates[0].is_file():
            raise RuntimeError(f"Saved detections not found: {candidates[0]}")
    else:
        candidates = [p for p in (root / "outputs").rglob("detections.csv") if p.parent.name == "owl-d"]
    best = ({}, dict(BASE), None, -1)
    for path in candidates:
        try:
            meta = read_json(path.parent / "metadata.json")
            config = read_json(path.parent.parent / "run_config.json")
            settings = config or meta.get("settings", {})
            source = config.get("video") or meta.get("source")
            if not source or Path(source).resolve() != video.resolve():
                continue
            if settings.get("source_size") not in {None, video.stat().st_size}:
                continue
            if settings.get("source_mtime_ns") not in {None, video.stat().st_mtime_ns}:
                continue
            profile = {key: settings.get(key, default) for key, default in BASE.items()}
            cached = cache_data(path, wanted, width, height)
            score = (len(cached), path.stat().st_mtime_ns)
            if score > (len(best[0]), best[3]):
                best = cached, profile, path, score[1]
        except (OSError, ValueError, KeyError, TypeError) as error:
            say(f"[CACHE SKIP] {path}: {error}")
    return best[:3]


def decoded_frames(video, first, stop, stride, width, height, output):
    from PIL import Image
    if shutil.which("ffmpeg"):
        log = (output / "decode.log").open("wb")
        filters = f"trim=start_frame={first}:end_frame={stop},select=not(mod(n\\,{stride}))"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2",
                   "-i", str(video), "-vf", filters, "-vsync", "0",
                   "-frames:v", str(len(range(first, stop, stride))), "-f", "rawvideo",
                   "-pix_fmt", "rgb24", "pipe:1"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
        complete = False
        try:
            size = width * height * 3
            for index in range(first, stop, stride):
                chunks, remaining = [], size
                while remaining:
                    chunk = process.stdout.read(remaining)
                    if not chunk:
                        raise RuntimeError(f"Decode ended before frame {index}; see {output / 'decode.log'}")
                    chunks.append(chunk)
                    remaining -= len(chunk)
                yield index, Image.frombytes("RGB", (width, height), b"".join(chunks))
            complete = True
        finally:
            process.stdout.close()
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                code = process.wait(timeout=10)
            log.close()
            if complete and code:
                raise RuntimeError(f"ffmpeg decoder exited with {code}; see decode.log")
    else:
        import cv2
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, first):
                raise RuntimeError("Cannot seek the source video")
            for index in range(first, stop, stride):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Video decoding failed at frame {index}")
                yield index, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for _ in range(min(stride-1, stop-index-1)):
                    if not cap.grab():
                        raise RuntimeError("Video decoding failed while skipping frames")
        finally:
            cap.release()


class Encoder:
    def __init__(self, path, width, height, fps):
        self.path, self.process, self.writer, self.log = path, None, None, None
        if shutil.which("ffmpeg"):
            codecs = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
            if "libx264" in codecs.stdout:
                self.log = path.with_suffix(".encode.log").open("wb")
                self.process = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                    "-r", f"{fps.numerator}/{fps.denominator}", "-i", "pipe:0", "-an",
                    "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "18",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=self.log)
        if self.process is None:
            import cv2
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
            if not self.writer.isOpened():
                raise RuntimeError("Neither H.264 nor OpenCV MP4 encoder is available")
            say("[ENCODER] OpenCV MPEG-4 fallback")

    def write(self, image):
        if self.process:
            try:
                self.process.stdin.write(image.tobytes())
            except BrokenPipeError as error:
                raise RuntimeError(f"Video encoder failed; see {self.path.with_suffix('.encode.log')}") from error
        else:
            import numpy as np
            self.writer.write(np.asarray(image)[:, :, ::-1].copy())

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.process is not None:
            process, self.process = self.process, None
            try:
                process.stdin.close()
                code = process.wait(timeout=30)
                if code:
                    raise RuntimeError(f"Video encoder exited with {code}")
            finally:
                self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/tahiti/AniCount"))
    parser.add_argument("--video", type=Path)
    parser.add_argument("--cache-run", help="Optional prior AniCount run folder")
    parser.add_argument("--cache-only", action="store_true", help="Require saved predictions for every selected frame")
    parser.add_argument("--start", type=float, default=61.)
    parser.add_argument("--end", type=float, default=68.)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--color", choices=["magenta", "cyan", "red", "yellow"], default="magenta")
    parser.add_argument("--numbers", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not (math.isfinite(args.start + args.end) and 0 <= args.start < args.end):
        parser.error("Choose finite 0 <= start < end")
    if args.stride < 1 or args.threads < 1 or not (3 <= args.radius <= 100 and 1 <= args.width <= args.radius):
        parser.error("Invalid stride, thread count, or ring dimensions")
    root = args.root.expanduser().resolve()
    video = args.video or root / "NG South Sudan 1920x1080 (1).m4v.mp4"
    video = (root / video if not video.is_absolute() else video).expanduser().resolve()
    if not video.is_file():
        raise RuntimeError(f"HD source not found: {video}")
    output = args.out or root / "outputs" / ("owl_d_rings_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    output = (root / output if not output.is_absolute() else output).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output folder is not empty: {output}. Choose a new --out.")
    say(f"[OUTPUT] {output / 'owl-d/annotated.mp4'}")
    marker_file = root / "highlight_owl.py"
    if not marker_file.is_file():
        marker_file = HERE / "highlight_owl.py"
    markers = load_module(marker_file, "_owl_video_markers")
    width, height, fps, total = video_info(video)
    if min(width, height, total) <= 0 or fps <= 0:
        raise RuntimeError("Invalid source video metadata")
    first = math.ceil(args.start * float(fps) - 1e-7)
    stop = min(total, math.ceil(args.end * float(fps) - 1e-7))
    indices = list(range(first, stop, args.stride))
    if not indices:
        raise RuntimeError("Selected interval is outside the source video")
    wanted = set(indices)
    cached, config, cache_path = choose_cache(root, args.cache_run, video, wanted, width, height)
    missing = len(indices) - len(cached)
    say(f"[SOURCE] {video}\n[VIDEO] {width}x{height} | {float(fps):.6f} FPS | {len(indices)} selected frames")
    say(f"[CACHE] {len(cached)}/{len(indices)} frames reusable | {cache_path or 'none'}")
    say(f"[INFERENCE] {missing} new frames | OWL-D | tile={config['tile']} overlap={config['overlap']} score={config['owl_conf']}")
    if args.cache_only and missing:
        raise RuntimeError(f"Saved predictions are missing for {missing} frames; omit --cache-only to calculate them")
    detector = device = None
    if missing:
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = str(args.threads)
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        backend = load_module(root / "anicount_video.py", "_owl_video_backend")
        device, reason = backend.select_device(args.device, 8.)
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        say(f"[DEVICE] {device} | {reason}")
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Selected CUDA is unavailable; use --device cpu")
        checkpoint = root / "weights/OWL-D.pth"
        if not checkpoint.is_file():
            raise RuntimeError(f"Existing weights not found: {checkpoint}")
        backend.verify_weights(checkpoint, "owl-d")
        detector = backend.Detector("owl-d", checkpoint, device, config)
    model_output = output / "owl-d"
    model_output.mkdir(parents=True, exist_ok=True)
    references = output / "reference_frames"
    references.mkdir()
    settings = {**config, "video": str(video), "start": args.start, "end": args.end,
                "stride": args.stride, "source_size": video.stat().st_size,
                "source_mtime_ns": video.stat().st_mtime_ns}
    write_json(output / "run_config.json", settings)
    meta = dict(model="owl-d", status="running", source=str(video), settings=settings,
                fps=float(fps), frames_processed=0, cached_frames=len(cached), device=device or "not used; cached detections")
    write_json(model_output / "metadata.json", meta)
    partial = model_output / "annotated.partial.mp4"
    sink = None
    decoder = decoded_frames(video, first, stop, args.stride, width, height, output)
    begin = time.monotonic()
    try:
        with (model_output / "detections.csv").open("w", newline="") as detections_file, \
             (model_output / "counts.csv").open("w", newline="") as counts_file:
            dw = csv.DictWriter(detections_file, fieldnames=DET_FIELDS)
            cw = csv.DictWriter(counts_file, fieldnames=["frame", "time_s", "count", "inference_s", "cached"])
            dw.writeheader()
            cw.writeheader()
            for number, (index, image) in enumerate(decoder, 1):
                seconds = index / float(fps)
                reused, elapsed = index in cached, 0.
                if reused:
                    detections = cached[index]
                else:
                    import numpy as np
                    say(f"[INFER] {number}/{len(indices)} | source {seconds:.3f}s")
                    detections, _, elapsed, _ = detector.predict(np.asarray(image)[:, :, ::-1].copy())
                styled = markers.render(image, detections, "owl-d", seconds, args.radius, args.width,
                                        markers.COLORS[args.color], args.numbers)
                if sink is None:
                    sink = Encoder(partial, styled.width, styled.height, fps / args.stride)
                sink.write(styled)
                for i, point in enumerate(detections, 1):
                    dw.writerow(dict(frame=index, time_s=round(seconds, 6), kind="point", detection_index=i,
                                     score=point["score"], x=point["x"], y=point["y"], category="animal"))
                cw.writerow(dict(frame=index, time_s=round(seconds, 6), count=len(detections),
                                 inference_s=round(elapsed, 4), cached=int(reused)))
                detections_file.flush()
                counts_file.flush()
                if number in {1, len(indices)} or abs(seconds-64.5) <= .5/float(fps):
                    stem = f"frame_{index:07d}_{seconds:010.3f}s.png"
                    styled.save(model_output / stem)
                    image.save(references / stem)
                meta.update(frames_processed=number, last_frame=index, last_count=len(detections))
                write_json(model_output / "metadata.json", meta)
                rate = number / max(time.monotonic()-begin, 1e-9)
                eta = (len(indices)-number) / max(rate, 1e-9)
                say(f"[{number}/{len(indices)}] t={seconds:.3f}s | count={len(detections)} | "
                    f"{'cached' if reused else f'inference {elapsed:.1f}s'} | ETA {eta/60:.1f} min")
        sink.close()
        sink = None
        final = model_output / "annotated.mp4"
        partial.replace(final)
        meta.update(status="complete", output_video=str(final))
        write_json(model_output / "metadata.json", meta)
        say(f"[DONE] {final}\n[CSV] {model_output / 'counts.csv'}")
        return 0
    except BaseException as error:
        meta.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        write_json(model_output / "metadata.json", meta)
        raise
    finally:
        decoder.close()
        if sink is not None:
            try:
                sink.close()
            except Exception as error:
                say(f"[ENCODER] {error}; CSV and partial video retained")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        say("\n[STOPPED] Saved per-frame coordinates and partial video retained.")
        raise SystemExit(130)
    except (OSError, RuntimeError, ValueError, ImportError, subprocess.SubprocessError) as error:
        say(f"[ERROR] {error}")
        raise SystemExit(1)
