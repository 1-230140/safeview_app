# SafeView Warehouse Scanner

The final detection path offers two models trained only from the 52 original warehouse
images and training-only derivatives:

- `models/fasterrcnn-warehouse-v3-best.pt`: Faster R-CNN ResNet50 FPN v2, the
  accuracy-focused option.
- `models/yolov10-warehouse-best.pt`: YOLOv10n, the speed-focused option.

Both recognize cardboard boxes, forklifts, safety cones, and safety vests. The former
YOLOv8 PPE model is not part of the app detection path.

Default login: `admin / safeview2025`.

## Run locally
on powershell
  cd "path file of the downloaded clone"
  1st powershell terminal
1. Install backend dependencies: `.venv\Scripts\python.exe -m pip install -r backend\requirements.txt`
2. Start the model API: `.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8001`
   2nd powershell terminal
3. In another terminal, serve this folder on port 8000: `.venv\Scripts\python.exe -m http.server 8000`
4. Open `http://localhost:8000`.

