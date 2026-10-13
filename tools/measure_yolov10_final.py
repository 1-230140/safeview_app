"""Generate the final YOLOv10 test evidence for the SafeView thesis.

Run this in the Kaggle session that contains the final YOLOv10 ``best.pt`` and
the untouched test split.  It writes Ultralytics' results.csv plus a compact
JSON report containing speed, peak GPU allocation, and host specifications.

Example:
    python tools/measure_yolov10_final.py \
        --model /kaggle/working/yolov10-warehouse-best.pt \
        --data /kaggle/working/warehouse_yolov10_2625/data.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import torch
from ultralytics import YOLO


def image_files(directory: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in extensions)


def hardware_details() -> dict[str, object]:
    details: dict[str, object] = {
        "operating_system": platform.platform(),
        "processor": platform.processor() or "not reported",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        details["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_gib": round(properties.total_memory / 1024**3, 2),
            "cuda_runtime": torch.version.cuda,
        }
    return details


def benchmark(model: YOLO, images: list[Path], device: str, repeats: int, warmup: int) -> dict[str, float]:
    if not images:
        raise ValueError("No test images were found for the speed measurement.")

    sample = str(images[0])
    for _ in range(warmup):
        model.predict(sample, imgsz=640, device=device, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    elapsed_seconds = 0.0
    measured_images = 0
    for _ in range(repeats):
        for image in images:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            model.predict(str(image), imgsz=640, device=device, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_seconds += time.perf_counter() - started
            measured_images += 1

    mean_ms = elapsed_seconds * 1000 / measured_images
    result = {
        "images_measured": measured_images,
        "warmup_runs": warmup,
        "mean_end_to_end_ms_per_image": round(mean_ms, 3),
        "fps": round(1000 / mean_ms, 3),
    }
    if torch.cuda.is_available():
        result["peak_gpu_memory_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
        result["peak_gpu_memory_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 1024**2, 2)
    return result


def copy_results_csv(run_directory: Path, output_directory: Path) -> Path | None:
    source = run_directory / "results.csv"
    if not source.exists():
        return None
    destination = output_directory / "yolov10-test-results.csv"
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Final YOLOv10 best.pt")
    parser.add_argument("--data", type=Path, required=True, help="YOLO data.yaml with test split")
    parser.add_argument("--output", type=Path, default=Path("yolov10-final-evidence"))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if not args.model.is_file() or not args.data.is_file():
        raise FileNotFoundError("Provide existing --model and --data paths.")

    args.output.mkdir(parents=True, exist_ok=True)
    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.model))  # .pt initializes from pretrained weights during training.

    # This is exactly the requested: yolo val model=best.pt data=data.yaml split=test.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    metrics = model.val(data=str(args.data), split="test", imgsz=640, device=device,
                        project=str(args.output), name="test", exist_ok=True, plots=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    test_images = image_files(args.data.parent / "images" / "test")
    report = {
        "model": str(args.model),
        "data": str(args.data),
        "split": "test",
        "input_size": [640, 640],
        "metrics": {
            "precision": round(float(metrics.box.mp), 6),
            "recall": round(float(metrics.box.mr), 6),
            "map50": round(float(metrics.box.map50), 6),
            "map50_95": round(float(metrics.box.map), 6),
        },
        "validation_peak_gpu_memory_mib": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 2) if torch.cuda.is_available() else None
        ),
        "speed": benchmark(model, test_images, device, args.repeats, args.warmup),
        "machine": hardware_details(),
    }

    csv_path = copy_results_csv(Path(metrics.save_dir), args.output)
    report["ultralytics_results_csv"] = str(csv_path) if csv_path else None
    report_path = args.output / "yolov10-final-summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
