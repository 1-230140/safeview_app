"""Compare final YOLOv10 and Faster R-CNN on six deterministic test scenarios.

The test split must remain untouched. Images are transformed in memory; their
ground-truth boxes receive the identical geometric rotation for the slant case.
The output is a micro precision/recall/F1 CSV, not mAP.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO


CLASS_NAMES = ["cardboard-box", "forklift", "safety-cone", "safety-vest"]
FASTER_RCNN_NAMES = {index + 1: name for index, name in enumerate(CLASS_NAMES)}


@dataclass
class Detection:
    label: str
    score: float
    box: list[float]


def iou(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def read_targets(label_path: Path, width: int, height: int) -> list[Detection]:
    targets: list[Detection] = []
    for row in label_path.read_text(encoding="utf-8").splitlines():
        class_id, cx, cy, box_width, box_height = map(float, row.split())
        class_id = int(class_id)
        x1 = (cx - box_width / 2) * width
        y1 = (cy - box_height / 2) * height
        x2 = (cx + box_width / 2) * width
        y2 = (cy + box_height / 2) * height
        targets.append(Detection(CLASS_NAMES[class_id], 1.0, [x1, y1, x2, y2]))
    return targets


def rotate(image: np.ndarray, targets: list[Detection]) -> tuple[np.ndarray, list[Detection]]:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 8, 1.0)
    rotated = cv2.warpAffine(image, matrix, (width, height), borderValue=(114, 114, 114))
    adjusted: list[Detection] = []
    for target in targets:
        x1, y1, x2, y2 = target.box
        corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float32)
        mapped = corners @ matrix.T
        new_box = [
            float(np.clip(mapped[:, 0].min(), 0, width)),
            float(np.clip(mapped[:, 1].min(), 0, height)),
            float(np.clip(mapped[:, 0].max(), 0, width)),
            float(np.clip(mapped[:, 1].max(), 0, height)),
        ]
        if new_box[2] > new_box[0] and new_box[3] > new_box[1]:
            adjusted.append(Detection(target.label, 1.0, new_box))
    return rotated, adjusted


def scenario(name: str, image: np.ndarray, targets: list[Detection]) -> tuple[np.ndarray, list[Detection]]:
    if name == "baseline_rgb":
        return image, targets
    if name == "slant_rotation":
        return rotate(image, targets)
    if name == "color_brightness":
        return cv2.convertScaleAbs(image, alpha=1.15, beta=25), targets
    if name == "grayscale":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), targets
    if name == "gaussian_blur":
        return cv2.GaussianBlur(image, (7, 7), 1.5), targets
    if name == "canny_edge":
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), targets
    raise ValueError(f"Unknown scenario: {name}")


def match(predictions: list[Detection], targets: list[Detection], threshold: float) -> tuple[int, int, int]:
    matches = 0
    used_targets: set[int] = set()
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        candidates = [
            (iou(prediction.box, target.box), index)
            for index, target in enumerate(targets)
            if index not in used_targets and target.label == prediction.label
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= threshold:
            matches += 1
            used_targets.add(best_index)
    false_positives = len(predictions) - matches
    false_negatives = len(targets) - matches
    return matches, false_positives, false_negatives


def load_faster_rcnn(path: Path, device: torch.device):
    model = fasterrcnn_resnet50_fpn_v2(weights=None, min_size=640, max_size=640)
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, len(CLASS_NAMES) + 1)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def faster_predictions(model, image: np.ndarray, confidence: float, device: torch.device) -> list[Detection]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device)
    with torch.inference_mode():
        result = model([tensor])[0]
    return [
        Detection(FASTER_RCNN_NAMES[int(label)], float(score), list(map(float, box)))
        for box, score, label in zip(result["boxes"].cpu().tolist(), result["scores"].cpu().tolist(), result["labels"].cpu().tolist())
        if float(score) >= confidence and int(label) in FASTER_RCNN_NAMES
    ]


def yolo_predictions(model: YOLO, image: np.ndarray, confidence: float, device: str) -> list[Detection]:
    result = model.predict(image, imgsz=640, conf=confidence, device=device, verbose=False)[0]
    if result.boxes is None:
        return []
    return [
        Detection(str(result.names[int(box.cls[0])]), float(box.conf[0]), list(map(float, box.xyxy[0].tolist())))
        for box in result.boxes
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root containing images/test and labels/test")
    parser.add_argument("--faster-rcnn", type=Path, required=True)
    parser.add_argument("--yolov10", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("six-scenario-comparison.csv"))
    parser.add_argument("--confidence", type=float, default=0.25, help="Same threshold for both models")
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()

    image_directory = args.dataset / "images" / "test"
    label_directory = args.dataset / "labels" / "test"
    images = sorted(path for path in image_directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise FileNotFoundError(f"No test images found in {image_directory}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_device = "0" if device.type == "cuda" else "cpu"
    faster = load_faster_rcnn(args.faster_rcnn, device)
    yolo = YOLO(str(args.yolov10))
    scenarios = ["baseline_rgb", "slant_rotation", "color_brightness", "grayscale", "gaussian_blur", "canny_edge"]
    rows: list[dict[str, object]] = []

    for scenario_name in scenarios:
        totals = {"faster_rcnn": [0, 0, 0], "yolov10": [0, 0, 0]}
        for image_path in images:
            image = cv2.imread(str(image_path))
            targets = read_targets(label_directory / f"{image_path.stem}.txt", image.shape[1], image.shape[0])
            image, targets = scenario(scenario_name, image, targets)
            for model_name, predictions in {
                "faster_rcnn": faster_predictions(faster, image, args.confidence, device),
                "yolov10": yolo_predictions(yolo, image, args.confidence, yolo_device),
            }.items():
                scores = match(predictions, targets, args.iou)
                totals[model_name] = [left + right for left, right in zip(totals[model_name], scores)]

        for model_name, (tp, fp, fn) in totals.items():
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append({"scenario": scenario_name, "model": model_name, "confidence": args.confidence,
                         "iou": args.iou, "tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4),
                         "recall": round(recall, 4), "f1": round(f1, 4)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved six-scenario comparison: {args.output}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
