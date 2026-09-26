import os
import time
import uuid
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import cv2
import numpy as np
import torch

from api.schemas import HealthResponse, ModelInfoResponse, PredictionResponse, Detection, BBox
from api.model_loader import RailGuardModelLoader

app = FastAPI(title="RailGuard Object Detection API")

# Add CORS Middleware for Render cross-origin communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        if file.content_type is not None:
            raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
            
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Corrupted image file.")

    # 2. Inference (CPU fallback safety for cloud deployment)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        results = model(image_np, conf=CONFIDENCE_THRESHOLD, device=device)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    
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

    # 3. MLOps Logging
    latency = time.time() - start_time
    info = loader.get_model_info()
    
    print(f"[INFERENCE LOG] model: {info['model_name']} | version: {info['version']} | alias: {info['alias']} | latency: {latency:.4f}s | detections: {len(detections)}")

    return PredictionResponse(detections=detections)

if __name__ == "__main__":
    import uvicorn
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
