#!/usr/bin/env python3
"""Render existing OWL points as prominent rings. No model loading or installs."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
import sys


COLORS = {"magenta": (255, 35, 180), "cyan": (0, 245, 255),
          "red": (255, 55, 55), "yellow": (255, 245, 0)}


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def choose_csv(root, requested, run, model):
    if requested:
        path = Path(requested).expanduser()
        return path if path.is_absolute() else root / path
    if run:
        directory = Path(run).expanduser()
        directory = directory if directory.is_absolute() else root / directory
        if directory.name != model:
            directory = directory / model
        return directory / "detections.csv"
    candidates = [p for p in (root / "outputs").rglob("detections.csv") if p.parent.name == model]
    if not candidates:
        raise RuntimeError(f"No saved {model} detections.csv under {root / 'outputs'}. Pass --csv PATH.")
    completed = [p for p in candidates if read_json(p.parent / "metadata.json").get("status") == "complete"]
    return max(completed or candidates, key=lambda p: p.stat().st_mtime_ns)


def select_frame(path, requested, at):
    frames = {}
    for row in rows(path):
        if not {"frame", "time_s", "x", "y"}.issubset(row):
            raise RuntimeError("CSV must contain frame,time_s,x,y from the AniCount run.")
        frames[int(row["frame"])] = float(row["time_s"])
    # counts.csv also includes frames with zero detections.
    counts = path.parent / "counts.csv"
    if counts.is_file():
        for row in rows(counts):
            frames[int(row["frame"])] = float(row["time_s"])
    if not frames:
        raise RuntimeError(f"No saved frames in {path}")
    index = requested if requested is not None else min(frames, key=lambda n: abs(frames[n] - at))
    if index not in frames:
        raise RuntimeError(f"Frame {index} is not present in this CSV")
    selected = [row for row in rows(path) if int(row["frame"]) == index]
    if any(row.get("kind", "point") != "point" for row in selected):
        raise RuntimeError("This CSV contains bbox predictions; choose the OWL-D or OWL-C CSV.")
    return index, frames[index], selected


def source_image(root, path, frame, explicit):
    from PIL import Image
    if explicit:
        image_path = Path(explicit).expanduser()
        image_path = image_path if image_path.is_absolute() else root / image_path
        with Image.open(image_path) as image:
            return image.convert("RGB"), str(image_path)
    run = path.parent.parent
    references = sorted((run / "reference_frames").glob(f"frame_{frame:07d}_*.png"))
    if references:
        with Image.open(references[0]) as image:
            return image.convert("RGB"), str(references[0])
    config = read_json(run / "run_config.json")
    metadata = read_json(path.parent / "metadata.json")
    video = config.get("video") or metadata.get("source")
    if not video:
        known = root / "NG South Sudan 1920x1080 (1).m4v.mp4"
        video = str(known) if known.is_file() else None
    if not video:
        raise RuntimeError("Original frame/video not found. Pass --image PATH to the original full-size frame.")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Saved original PNG is missing. Use an existing Python with cv2, or pass --image PATH.") from error
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, frame):
            raise RuntimeError(f"Cannot read frame {frame} from {video}")
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot decode frame {frame} from {video}")
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), str(video)
    finally:
        cap.release()


def render(image, detections, model, seconds, radius, width, color, numbers):
    from PIL import Image, ImageDraw, ImageFont
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font_size = max(12, min(26, image.width // 42))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    for i, row in enumerate(detections, 1):
        x, y = float(row["x"]), float(row["y"])
        if not math.isfinite(x + y) or not (0 <= x < image.width and 0 <= y < image.height):
            raise RuntimeError("Point lies outside the source image; use the original frame at its native resolution.")
        x, y = int(round(x)), int(round(y))
        r = radius
        # Dark halo, colored hollow ring, and a small white center.
        draw.ellipse((x-r-2, y-r-2, x+r+2, y+r+2), outline=(0, 0, 0), width=width+4)
        draw.ellipse((x-r, y-r, x+r, y+r), outline=color, width=width)
        draw.ellipse((x-1, y-1, x+1, y+1), fill=(255, 255, 255))
        if numbers:
            label = str(row.get("detection_index") or i)
            draw.text((x+r+3, y-r), label, fill=(255, 255, 255), font=font,
                      stroke_width=2, stroke_fill=(0, 0, 0))
    footer_height = max(40, font_size + 20)
    output = Image.new("RGB", (result.width, result.height + footer_height), (15, 18, 23))
    output.paste(result, (0, 0))
    ImageDraw.Draw(output).text((12, result.height+8),
        f"{model.upper()} | count={len(detections)} | source {seconds:.3f}s",
        fill=(245, 245, 245), font=font)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/home/tahiti/AniCount"))
    p.add_argument("--run", help="Existing run folder, e.g. outputs/probe_645; default: latest completed OWL run")
    p.add_argument("--csv", help="Explicit existing detections.csv")
    p.add_argument("--model", choices=["owl-d", "owl-c"], default="owl-d")
    p.add_argument("--at", type=float, default=64.5, help="Select the saved frame closest to this source time")
    p.add_argument("--frame", type=int, help="Select this source frame index instead of --at")
    p.add_argument("--image", help="Optional original full-size frame PNG/JPG")
    p.add_argument("--radius", type=int, default=10, help="Ring radius in native image pixels")
    p.add_argument("--width", type=int, default=3)
    p.add_argument("--color", choices=list(COLORS), default="magenta")
    p.add_argument("--numbers", action="store_true", help="Add detection indices; can clutter dense herds")
    p.add_argument("--out", type=Path, help="New output PNG; default: inside the model's highlighted folder")
    args = p.parse_args()
    if not (3 <= args.radius <= 100 and 1 <= args.width <= args.radius):
        p.error("Choose radius 3..100 and width 1..radius")
    if not math.isfinite(args.at):
        p.error("--at must be finite")
    root = args.root.expanduser().resolve()
    csv_path = choose_csv(root, args.csv, args.run, args.model)
    if not csv_path.is_file():
        raise RuntimeError(f"CSV not found: {csv_path}")
    index, seconds, detections = select_frame(csv_path, args.frame, args.at)
    image, source = source_image(root, csv_path, index, args.image)
    output = render(image, detections, args.model, seconds, args.radius, args.width,
                    COLORS[args.color], args.numbers)
    filename = f"frame_{index:07d}_{seconds:010.3f}s_{dt.datetime.now():%Y%m%d_%H%M%S_%f}.png"
    target = args.out or csv_path.parent / "highlighted" / filename
    if not target.is_absolute():
        target = root / target
    if target.exists():
        raise RuntimeError(f"Output already exists; choose another --out: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    output.save(target, format="PNG")
    print(f"[CSV] {csv_path}\n[SOURCE] {source}\n[FRAME] {index} | {seconds:.3f}s\n"
          f"[COUNT] {len(detections)} saved points\n[OUTPUT] {target}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print(f"[ERROR] {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
