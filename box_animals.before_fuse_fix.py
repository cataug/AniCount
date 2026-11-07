import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from torchvision.ops import nms
from ultralytics import YOLOWorld


def starts(length, tile, overlap):
    if length <= tile:
        return [0]
    step = tile - overlap
    return sorted(set(
        list(range(0, length - tile + 1, step)) + [length - tile]
    ))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--out", default="outputs")
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--tile", type=int, default=768)
    p.add_argument("--overlap", type=int, default=256)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--iou", type=float, default=0.45)
    args = p.parse_args()

    if not 0 <= args.overlap < args.tile:
        p.error("Require 0 <= overlap < tile")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")

    src = Path(args.video).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Invalid video metadata")

    target = out / f"{src.stem}_boxed.mp4"
    annotations = out / f"{src.stem}_boxes.jsonl"
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {target}")

    # Prompts aid detection; their output is not a reliable species ID.
    classes = [
        "antelope", "gazelle", "deer", "wildebeest",
        "buffalo", "cow", "elephant", "giraffe", "zebra",
        "hippopotamus", "lion",
    ]

    print("Loading YOLO-World; first run downloads weights.", flush=True)
    model = YOLOWorld("yolov8x-worldv2.pt")
    model.to("cuda:0")
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection

    clip_id = "openai/clip-vit-base-patch32"
    print("Loading CLIP from Hugging Face cache...", flush=True)
    try:
        tokenizer = CLIPTokenizer.from_pretrained(
            clip_id, local_files_only=True
        )
        encoder = CLIPTextModelWithProjection.from_pretrained(
            clip_id, local_files_only=True
        )
    except OSError:
        print("Cache incomplete; downloading CLIP from Hugging Face.", flush=True)
        tokenizer = CLIPTokenizer.from_pretrained(clip_id)
        encoder = CLIPTextModelWithProjection.from_pretrained(clip_id)

    encoder.eval()
    tokens = tokenizer(
        classes, padding=True, truncation=True,
        max_length=77, return_tensors="pt"
    )
    with torch.no_grad():
        features = encoder(**tokens).text_embeds
        features = torch.nn.functional.normalize(features, dim=-1)

    device = next(model.model.parameters()).device
    model.model.txt_feats = features.unsqueeze(0).to(device)
    model.model.model[-1].nc = len(classes)
    model.model.names = list(classes)
    model.predictor = None
    del encoder, tokenizer, tokens, features
    print("CLIP ready. Starting video processing.", flush=True)

    print(
        f"GPU: {torch.cuda.get_device_name(0)}\n"
        f"Video: {width}x{height}, {fps:.3f} FPS, {total} frames",
        flush=True,
    )

    index = 0
    began = time.monotonic()

    try:
        with annotations.open("w", encoding="utf-8") as log:
            with torch.inference_mode():
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    # Whole frame plus overlapping crops.
                    views = [frame]
                    offsets = [(0, 0)]
                    if width > args.tile or height > args.tile:
                        for y in starts(height, args.tile, args.overlap):
                            for x in starts(width, args.tile, args.overlap):
                                views.append(
                                    frame[y:y + args.tile, x:x + args.tile]
                                )
                                offsets.append((x, y))

                    candidates = []
                    for start in range(0, len(views), args.batch):
                        results = model.predict(
                            source=views[start:start + args.batch],
                            imgsz=args.imgsz,
                            conf=args.conf,
                            iou=args.iou,
                            agnostic_nms=True,
                            max_det=10000,
                            device=0,
                            half=True,
                            verbose=False,
                        )
                        for result, (x, y) in zip(
                            results, offsets[start:start + args.batch]
                        ):
                            for row in result.boxes.data.cpu().tolist():
                                x1, y1, x2, y2, score, category = row
                                candidates.append([
                                    x1 + x, y1 + y, x2 + x, y2 + y,
                                    score, int(category),
                                ])

                    detections = []
                    if candidates:
                        data = torch.tensor(candidates, dtype=torch.float32)
                        data[:, [0, 2]].clamp_(0, width)
                        data[:, [1, 3]].clamp_(0, height)

                        # Class-agnostic merge: one animal may match
                        # several prompts or occur in overlapping crops.
                        keep = nms(data[:, :4], data[:, 4], args.iou)
                        for row in data[keep].tolist():
                            x1, y1, x2, y2, score, category = row
                            detections.append({
                                "xyxy": [round(v, 2) for v in row[:4]],
                                "confidence": round(score, 4),
                                "prompt": classes[int(category)],
                            })
                            cv2.rectangle(
                                frame,
                                (round(x1), round(y1)),
                                (round(x2), round(y2)),
                                (60, 255, 70), 1,
                            )

                    cv2.rectangle(frame, (8, 8), (410, 46), (20, 20, 20), -1)
                    cv2.putText(
                        frame,
                        f"Detected: {len(detections)} | Frame: {index}",
                        (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2,
                    )
                    writer.write(frame)
                    log.write(json.dumps({
                        "frame": index,
                        "time_s": round(index / fps, 6),
                        "count": len(detections),
                        "detections": detections,
                    }) + "\n")

                    index += 1
                    if index == 1 or index % 30 == 0:
                        elapsed = time.monotonic() - began
                        print(
                            f"{index}/{total} frames | "
                            f"{len(detections)} detections | "
                            f"{index / elapsed:.2f} processing FPS",
                            flush=True,
                        )
    finally:
        cap.release()
        writer.release()

    print(f"Completed: {index} frames\n{target}\n{annotations}", flush=True)


if __name__ == "__main__":
    main()
