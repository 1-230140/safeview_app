# SafeView PPE Scanner

Open `login.html` from a local web server. Camera access usually requires `https://` or `localhost`.

Default login: `admin / safeview2025`.

Model: `models/ppe-yolo.onnx`, copied from your `best.onnx`.

Roboflow API is not included because this app uses your local ONNX model and does not need an API key.

To train again with your labeled YOLO dataset:

```bash
pip install ultralytics
yolo detect train model=yolov8n.pt data=data.yaml imgsz=640 epochs=100
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640
```

Replace `models/ppe-yolo.onnx` with the exported ONNX file. Make sure `CLASS_NAMES` in `app.js` matches your `data.yaml` names order.
