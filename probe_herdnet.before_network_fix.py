from pathlib import Path
import hashlib
import json
import subprocess
import sys
import urllib.request
import zipfile

import cv2
import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parent
out = root / "probe_herdnet"
vendor = root / "vendor" / "herdnet_minimal"
out.mkdir(parents=True, exist_ok=True)
vendor.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)

source = root / "probe_original" / "crop_original.png"
image = cv2.imread(str(source))
if image is None:
    raise RuntimeError(f"Не удалось прочитать исходный кроп: {source}")

def digest(path, algorithm):
    if not path.is_file():
        return ""
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

# Download the distribution without installing its dependencies.
print("Подготовка архитектуры HerdNet...", flush=True)
metadata_url = "https://pypi.org/pypi/PytorchWildlife/1.3.1/json"
with urllib.request.urlopen(metadata_url, timeout=60) as response:
    metadata = json.load(response)

wheel_info = next(
    item for item in metadata["urls"]
    if item["filename"].endswith(".whl")
)
wheel = vendor / wheel_info["filename"]
if digest(wheel, "sha256") != wheel_info["digests"]["sha256"]:
    with urllib.request.urlopen(wheel_info["url"], timeout=60) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != wheel_info["digests"]["sha256"]:
        raise RuntimeError("Контрольная сумма архива не совпала.")
    wheel.write_bytes(payload)

# Minimal package: architecture only, without package-wide imports.
package_parts = [
    "PytorchWildlife",
    "PytorchWildlife/models",
    "PytorchWildlife/models/detection",
    "PytorchWildlife/models/detection/localization",
]
for part in package_parts:
    directory = vendor / part
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "__init__.py").write_text("")

with zipfile.ZipFile(wheel) as archive:
    for filename in ("dla.py", "model.py"):
        suffix = "/models/detection/localization/" + filename
        matches = [
            name for name in archive.namelist() if name.endswith(suffix)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"В архиве не найден единственный {filename}")
        destination = vendor / package_parts[-1] / filename
        destination.write_bytes(archive.read(matches[0]))

    licenses = [
        name for name in archive.namelist()
        if "license" in name.lower() and not name.endswith("/")
    ]
    for number, name in enumerate(licenses):
        (vendor / f"LICENSE_{number}.txt").write_bytes(archive.read(name))

sys.path.insert(0, str(vendor))
from PytorchWildlife.models.detection.localization.model import HerdNet

weights = root / "weights" / "herdnet_general.pth"
weights.parent.mkdir(exist_ok=True)
expected_md5 = "fcc301fcb61d57579764891247de4b7b"

if digest(weights, "md5") != expected_md5:
    print("Скачивание весов HerdNet — около 220 МБ.", flush=True)
    partial = weights.with_suffix(".pth.part")
    url = (
        "https://zenodo.org/records/18177050/files/"
        "20220413_HerdNet_General_dataset_2022.pth?download=1"
    )
    subprocess.run([
        "curl", "--location", "--fail",
        "--retry", "2", "--connect-timeout", "20",
        "--max-time", "900", "--continue-at", "-",
        "--output", str(partial), url,
    ], check=True)
    if digest(partial, "md5") != expected_md5:
        raise RuntimeError("Контрольная сумма весов не совпала.")
    partial.replace(weights)

# Checkpoint comes from the verified research-model download above.
checkpoint = torch.load(
    weights, map_location="cpu", weights_only=False
)
model = HerdNet(
    num_classes=len(checkpoint["classes"]) + 1,
    pretrained=False,
)
state = checkpoint["model_state_dict"]
if any(key.startswith("model.") for key in state):
    state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
model.load_state_dict(state, strict=True)
model.eval()

# Enlarge this 256-pixel crop to 512 pixels.
h, w = image.shape[:2]
resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
mean = torch.as_tensor(checkpoint["mean"], dtype=torch.float32).view(3, 1, 1)
std = torch.as_tensor(checkpoint["std"], dtype=torch.float32).view(3, 1, 1)
tensor = ((tensor - mean) / std).unsqueeze(0)

print("Распознавание одного кропа на CPU...", flush=True)
with torch.inference_mode():
    heatmap, _ = model(tensor)

# Diagnostic localization peaks; no species filtering.
heatmap = heatmap[0, 0]
local_maxima = F.max_pool2d(
    heatmap[None, None], kernel_size=3, stride=1, padding=1
)[0, 0]
peaks = heatmap == local_maxima
print(f"Максимум карты обнаружений: {heatmap.max().item():.4f}",
      flush=True)

for threshold in (0.10, 0.20, 0.30):
    locations = torch.nonzero(
        peaks & (heatmap >= threshold), as_tuple=False
    )
    canvas = cv2.resize(image, (768, 768),
                        interpolation=cv2.INTER_LINEAR)
    points = []
    for row, column in locations.tolist():
        x = column * w / heatmap.shape[1]
        y = row * h / heatmap.shape[0]
        points.append({"x": x, "y": y,
                       "score": float(heatmap[row, column])})
        position = (round(x * 768 / w), round(y * 768 / h))
        cv2.circle(canvas, position, 4, (0, 0, 0), -1)
        cv2.circle(canvas, position, 2, (40, 255, 40), -1)

    canvas = cv2.copyMakeBorder(
        canvas, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25)
    )
    cv2.putText(
        canvas, f"HerdNet | threshold={threshold:.2f} | peaks={len(points)}",
        (10, 795), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 1,
    )
    tag = f"{round(threshold * 100):03d}"
    path = out / f"centers_{tag}.png"
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Не удалось сохранить {path}")
    (out / f"centers_{tag}.json").write_text(json.dumps(points, indent=2))
    print(f"{len(points)} кандидатов: {path}", flush=True)
