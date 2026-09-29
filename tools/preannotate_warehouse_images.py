"""Create reviewable YOLO pre-annotations from the current SafeView YOLOv8 model.

The generated labels are suggestions only. Review every box before training.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


INPUT_SIZE = 640
CONFIDENCE = 0.10
IOU_THRESHOLD = 0.45

# Original SafeView YOLOv8 index -> warehouse dataset index and class name.
TARGETS = {
    5: (0, "forklift"),
    7: (1, "cardboard-box"),
    6: (2, "safety-vest"),
    8: (3, "safety-cone"),
}
COLORS = {
    "forklift": "#d14b4b",
    "cardboard-box": "#815d35",
    "safety-vest": "#168a4a",
    "safety-cone": "#b46b00",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jpe", ".png"}


def letterbox(image: Image.Image) -> tuple[np.ndarray, float, float, float]:
    width, height = image.size
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    new_width, new_height = round(width * scale), round(height * scale)
    dx, dy = (INPUT_SIZE - new_width) / 2, (INPUT_SIZE - new_height) / 2
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), "#727272")
    canvas.paste(image.resize((new_width, new_height)), (round(dx), round(dy)))
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.transpose(pixels, (2, 0, 1))[None], scale, dx, dy


def iou(first: dict, second: dict) -> float:
    x1, y1 = max(first["x1"], second["x1"]), max(first["y1"], second["y1"])
    x2, y2 = min(first["x2"], second["x2"]), min(first["y2"], second["y2"])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = (first["x2"] - first["x1"]) * (first["y2"] - first["y1"])
    second_area = (second["x2"] - second["x1"]) * (second["y2"] - second["y1"])
    return intersection / max(first_area + second_area - intersection, 1e-9)


def nms(boxes: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for box in sorted(boxes, key=lambda item: item["score"], reverse=True):
        if all(iou(box, selected) < IOU_THRESHOLD for selected in kept):
            kept.append(box)
    return kept


def predict(session: ort.InferenceSession, image: Image.Image) -> list[dict]:
    tensor, scale, dx, dy = letterbox(image)
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    predictions = output.shape[-1]
    width, height = image.size
    boxes = []

    for index in range(predictions):
        source_class, score = max(
            ((class_id, float(output[0, 4 + class_id, index])) for class_id in TARGETS),
            key=lambda item: item[1],
        )
        if score < CONFIDENCE:
            continue
        class_id, class_name = TARGETS[source_class]
        center_x, center_y = float(output[0, 0, index]), float(output[0, 1, index])
        box_width, box_height = float(output[0, 2, index]), float(output[0, 3, index])
        x1 = max(0.0, min(width, (center_x - box_width / 2 - dx) / scale))
        y1 = max(0.0, min(height, (center_y - box_height / 2 - dy) / scale))
        x2 = max(0.0, min(width, (center_x + box_width / 2 - dx) / scale))
        y2 = max(0.0, min(height, (center_y + box_height / 2 - dy) / scale))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({
            "class_id": class_id,
            "class_name": class_name,
            "score": score,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

    grouped = {}
    for box in boxes:
        grouped.setdefault(box["class_id"], []).append(box)
    return [box for group in grouped.values() for box in nms(group)]


def save_yolo_label(path: Path, boxes: list[dict], width: int, height: int) -> None:
    rows = []
    for box in boxes:
        center_x = ((box["x1"] + box["x2"]) / 2) / width
        center_y = ((box["y1"] + box["y2"]) / 2) / height
        box_width = (box["x2"] - box["x1"]) / width
        box_height = (box["y2"] - box["y1"]) / height
        rows.append(f"{box['class_id']} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
    path.write_text("\n".join(rows), encoding="utf-8")


def save_preview(path: Path, image: Image.Image, boxes: list[dict]) -> None:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for box in boxes:
        color = COLORS[box["class_name"]]
        draw.rectangle((box["x1"], box["y1"], box["x2"], box["y2"]), outline=color, width=4)
        draw.text((box["x1"] + 4, max(0, box["y1"] - 18)), f"{box['class_name']} {box['score']:.0%}", fill=color)
    preview.save(path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    images = sorted(path for path in args.source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if len(images) != 52:
        raise RuntimeError(f"Expected 52 images, found {len(images)}.")

    image_dir = args.output / "images" / "all"
    label_dir = args.output / "labels-yolo" / "all"
    preview_dir = args.output / "review-previews"
    for folder in (image_dir, label_dir, preview_dir):
        folder.mkdir(parents=True, exist_ok=True)

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    rows = []
    for number, source_path in enumerate(images, start=1):
        destination_name = f"warehouse_{number:03d}.jpg"
        destination_image = image_dir / destination_name
        with Image.open(source_path) as source_image:
            image = source_image.convert("RGB")
            image.save(destination_image, quality=95)
            boxes = predict(session, image)
            save_yolo_label(label_dir / f"warehouse_{number:03d}.txt", boxes, *image.size)
            save_preview(preview_dir / destination_name, image, boxes)
        rows.append({
            "image": destination_name,
            "source": source_path.name,
            "preannotated_boxes": len(boxes),
            "classes": ", ".join(sorted({box['class_name'] for box in boxes})) or "none",
            "review_required": "yes",
        })
        print(f"{number:02d}/52 {source_path.name}: {len(boxes)} suggested box(es)")

    with (args.output / "review-manifest.csv").open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "classes.txt").write_text("forklift\ncardboard-box\nsafety-vest\nsafety-cone\n", encoding="utf-8")
    print(f"\nCreated review dataset: {args.output}")
    print("Review every preview and correct its matching YOLO label before training.")


if __name__ == "__main__":
    main()
