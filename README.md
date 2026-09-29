# SafeView Warehouse Scanner

The final detection path uses `models/fasterrcnn-warehouse-v3-best.pt`: Faster R-CNN
ResNet50 FPN v2 trained only from the 52 original warehouse images and training-only
derivatives. It recognizes cardboard boxes, forklifts, safety cones, and safety vests.

Default login: `admin / safeview2025`.

## Run locally

1. Install backend dependencies: `.venv\Scripts\python.exe -m pip install -r backend\requirements.txt`
2. Start the model API: `.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8001`
3. In another terminal, serve this folder on port 8000: `.venv\Scripts\python.exe -m http.server 8000`
4. Open `http://localhost:8000`.

The local model checkpoint is intentionally ignored by Git. This is an academic prototype;
its five-image test evaluation is too small for operational safety decisions.
