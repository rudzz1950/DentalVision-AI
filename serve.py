import os
import io
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from inference import ToothDetector

# Load model configuration from environment variables (provide sensible defaults)
MODEL_PATH = os.getenv("MODEL_PATH", "runs/train/exp/weights/best.pt")
DATA_YAML = os.getenv("DATA_YAML", "ToothNumber_TaskDataset/dental_teeth.yaml")

# Initialize detector once
_detector: Optional[ToothDetector] = None


def get_detector() -> ToothDetector:
    global _detector
    if _detector is None:
        _detector = ToothDetector(MODEL_PATH, data_yaml_path=DATA_YAML)
    return _detector


class Detection(BaseModel):
    class_id: int
    fdi: int
    confidence: float
    # Normalized coordinates [0,1]
    x1: float
    y1: float
    x2: float
    y2: float


class PredictResponse(BaseModel):
    num_detections: int
    detections: List[Detection]


app = FastAPI(title="Dental Teeth Detection API", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    conf: float = Form(0.5),
    iou: float = Form(0.5),
    tta: bool = Form(False),
    soft_nms: bool = Form(False),
    soft_nms_method: str = Form("gaussian"),
    soft_nms_sigma: float = Form(0.5),
):
    detector = get_detector()
    # Read image bytes and decode to BGR
    image_bytes = await file.read()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Run model
    results = detector.model(
        img_rgb,
        conf=conf,
        iou=iou,
        augment=tta,
        verbose=False,
    )
    det = results[0]

    # Optional Soft-NMS
    if soft_nms and len(det.boxes) > 0:
        detector.apply_soft_nms(det, iou_thresh=iou, sigma=soft_nms_sigma,
                                method=soft_nms_method, conf_filter=conf)

    detections: List[Detection] = []
    if len(det.boxes) > 0:
        xyxyn = det.boxes.xyxyn.cpu().numpy()
        cls = det.boxes.cls.cpu().numpy().astype(int)
        confs = det.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, s in zip(xyxyn, cls, confs):
            detections.append(
                Detection(
                    class_id=int(c),
                    fdi=int(detector.class_to_fdi.get(int(c), int(c))),
                    confidence=float(s),
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                )
            )

    return PredictResponse(num_detections=len(detections), detections=detections)


@app.post("/predict-batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    conf: float = Form(0.5),
    iou: float = Form(0.5),
    tta: bool = Form(False),
    soft_nms: bool = Form(False),
    soft_nms_method: str = Form("gaussian"),
    soft_nms_sigma: float = Form(0.5),
):
    detector = get_detector()
    responses = []
    for file in files:
        image_bytes = await file.read()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            responses.append({"filename": file.filename, "error": "Invalid image"})
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = detector.model(
            img_rgb,
            conf=conf,
            iou=iou,
            augment=tta,
            verbose=False,
        )
        det = results[0]
        if soft_nms and len(det.boxes) > 0:
            detector.apply_soft_nms(det, iou_thresh=iou, sigma=soft_nms_sigma,
                                    method=soft_nms_method, conf_filter=conf)
        detections = []
        if len(det.boxes) > 0:
            xyxyn = det.boxes.xyxyn.cpu().numpy()
            cls = det.boxes.cls.cpu().numpy().astype(int)
            confs = det.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, s in zip(xyxyn, cls, confs):
                detections.append({
                    "class_id": int(c),
                    "fdi": int(detector.class_to_fdi.get(int(c), int(c))),
                    "confidence": float(s),
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                })
        responses.append({
            "filename": file.filename,
            "num_detections": len(detections),
            "detections": detections,
        })
    return {"results": responses}
