import os
import time
import uuid
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import cv2
import numpy as np

from api.schemas import HealthResponse, ModelInfoResponse, PredictionResponse, Detection, BBox
from api.model_loader import RailGuardModelLoader

app = FastAPI(title="RailGuard Object Detection API")

# Load Configuration
with open("configs/serve.yaml", "r") as f:
    config = yaml.safe_load(f)

CONFIDENCE_THRESHOLD = config["inference"]["confidence_threshold"]
OUTPUT_DIR = config["inference"]["output_dir"]

# Initialize Model Loader
loader = RailGuardModelLoader()
model = None

@app.on_event("startup")
def startup_event():
    global model
    model = loader.load_model()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/health", response_model=HealthResponse)
def health_check():
    info = loader.get_model_info()
    return HealthResponse(
        status="healthy",
        service="RailGuard",
        model=info["model_name"],
        alias=info["alias"]
    )

@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    info = loader.get_model_info()
    return ModelInfoResponse(**info)

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    # 1. Image Validation
    if not file.content_type or not file.content_type.startswith("image/"):
        # We can also fallback to checking extension or just proceed and let PIL fail gracefully,
        # but for safety we will just let it proceed if content_type is None, or throw if it's explicitly not image.
        if file.content_type is not None:
            raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
            
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        # Convert RGB to BGR for OpenCV processing if needed, but ultralytics handles RGB well
    except Exception as e:
        raise HTTPException(status_code=400, detail="Corrupted image file.")

    # 2. Inference
    results = model(image_np, conf=CONFIDENCE_THRESHOLD)
    
    detections = []
    
    if len(results) > 0:
        result = results[0]
        
        # Save annotated image
        annotated_img = result.plot()
        output_filename = f"pred_{uuid.uuid4().hex[:8]}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        
        # Parse boxes
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=BBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
                )
            )

    # 3. MLOps Logging (printing to console serves as basic structured logging)
    latency = time.time() - start_time
    info = loader.get_model_info()
    
    print(f"[INFERENCE LOG] model: {info['model_name']} | version: {info['version']} | alias: {info['alias']} | latency: {latency:.4f}s | detections: {len(detections)}")

    return PredictionResponse(detections=detections)
