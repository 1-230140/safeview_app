from io import BytesIO
from pathlib import Path
from time import perf_counter

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
FASTER_RCNN_PATH = ROOT / "models" / "fasterrcnn-warehouse-v3-best.pt"
YOLOV10_PATH = ROOT / "models" / "yolov10-warehouse-best.pt"
CLASS_NAMES = {1: "cardboard-box", 2: "forklift", 3: "safety-cone", 4: "safety-vest"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="SafeView Warehouse Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

faster_rcnn_model = None
yolov10_model = None


def load_faster_rcnn():
    global faster_rcnn_model
    if faster_rcnn_model is not None:
        return faster_rcnn_model
    if not FASTER_RCNN_PATH.exists():
        raise RuntimeError(f"Model checkpoint not found: {FASTER_RCNN_PATH}")

    detector = fasterrcnn_resnet50_fpn_v2(weights=None, min_size=640, max_size=640)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES) + 1)
    checkpoint = torch.load(FASTER_RCNN_PATH, map_location=DEVICE, weights_only=False)
    detector.load_state_dict(checkpoint["model_state_dict"])
    detector.to(DEVICE).eval()
    faster_rcnn_model = detector
    return faster_rcnn_model


def load_yolov10():
    global yolov10_model
    if yolov10_model is not None:
        return yolov10_model
    if not YOLOV10_PATH.exists():
        raise RuntimeError(f"Model checkpoint not found: {YOLOV10_PATH}")
    yolov10_model = YOLO(str(YOLOV10_PATH))
    return yolov10_model


@app.get("/health")
def health():
    return {
        "status": "ready",
        "models": {
            "fasterrcnn": FASTER_RCNN_PATH.exists(),
            "yolov10": YOLOV10_PATH.exists(),
        },
        "classes": list(CLASS_NAMES.values()),
        "input_size": [640, 640],
        "device": str(DEVICE),
    }


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    confidence: float = Query(default=0.70, ge=0.05, le=0.95),
    model: str = Query(default="fasterrcnn", pattern="^(fasterrcnn|yolov10)$"),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")

    try:
        original = Image.open(BytesIO(await image.read())).convert("RGB")
    except Exception as error:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from error

    started = perf_counter()
    if model == "fasterrcnn":
        tensor = pil_to_tensor(original).float().div(255).to(DEVICE)
        with torch.inference_mode():
            output = load_faster_rcnn()([tensor])[0]
        detections = faster_rcnn_detections(output, confidence)
        model_name = "Faster R-CNN v3"
    else:
        results = load_yolov10().predict(
            source=original,
            imgsz=640,
            conf=confidence,
            device=0 if DEVICE.type == "cuda" else "cpu",
            verbose=False,
        )
        detections = yolov10_detections(results[0])
        model_name = "YOLOv10n"
    elapsed_ms = (perf_counter() - started) * 1000

    return {
        "detections": detections,
        "inference_ms": round(elapsed_ms, 1),
        "input_width": original.width,
        "input_height": original.height,
        "confidence": confidence,
        "model": model,
        "model_name": model_name,
    }


def faster_rcnn_detections(output, confidence):
    detections = []
    for box, score, label in zip(output["boxes"].cpu(), output["scores"].cpu(), output["labels"].cpu()):
        score_value = float(score.item())
        label_value = int(label.item())
        if score_value < confidence or label_value not in CLASS_NAMES:
            continue
        x1, y1, x2, y2 = (float(value) for value in box.tolist())
        detections.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "score": score_value,
            "label": CLASS_NAMES[label_value],
        })

    return detections


def yolov10_detections(result):
    detections = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].cpu().tolist())
        class_id = int(box.cls[0].item())
        detections.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "score": float(box.conf[0].item()),
            "label": str(result.names[class_id]),
        })
    return detections
