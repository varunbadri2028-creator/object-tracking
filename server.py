import os
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# Vercel expects a top-level FastAPI instance named `app` in the entrypoint file.
app = FastAPI(title="Object Tracking API (YOLOv8)")

MODEL_PATH = os.getenv("MODEL", "yolov8n.pt")
CONF = float(os.getenv("CONF", "0.25"))

_obj_model = None


def get_model():
    global _obj_model
    if _obj_model is None:
        _obj_model = YOLO(MODEL_PATH)
    return _obj_model


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/detect")
def detect(file: UploadFile = File(...)):
    data = file.file.read()

    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    model = get_model()
    results = model.predict(tmp_path, conf=CONF, verbose=False)

    detections = []
    if results and len(results) > 0:
        r = results[0]
        if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                detections.append(
                    {
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "conf": float(box.conf[0].item()),
                        "class": int(box.cls[0].item()),
                    }
                )

    return JSONResponse({"file": file.filename, "detections": detections})

