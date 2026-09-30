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

from prometheus_client import make_asgi_app, Counter, Histogram, Gauge

# Optimize memory usage for Render Free Tier
torch.set_num_threads(1)

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

# --- PROMETHEUS METRICS ---
PREDICTION_REQUESTS_TOTAL = Counter("railguard_prediction_requests_total", "Total prediction requests", ["status"])
HTTP_REQUESTS_TOTAL = Counter("railguard_http_requests_total", "Total HTTP requests", ["endpoint", "method", "status_code"])
PREDICTION_LATENCY = Histogram("railguard_prediction_latency_seconds", "Total prediction request latency")
INFERENCE_LATENCY = Histogram("railguard_inference_latency_seconds", "Model inference execution latency")
DETECTIONS_TOTAL = Counter("railguard_detections_total", "Total detected defects by class", ["class_name"])
DETECTIONS_PER_REQUEST = Histogram("railguard_detections_per_request", "Number of defects detected per request")
MODEL_INFO = Gauge("railguard_model_info", "Currently loaded model information", ["model_name", "version", "alias"])
API_HEALTH = Gauge("railguard_api_health", "API health status (1=online)")

# Expose metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
# --------------------------

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
    
    # Init Prometheus Gauges
    API_HEALTH.set(1)
    info = loader.get_model_info()
    MODEL_INFO.labels(model_name=info["model_name"], version=info["version"], alias=info["alias"]).set(1)

@app.get("/health", response_model=HealthResponse)
def health_check():
    HTTP_REQUESTS_TOTAL.labels(endpoint="/health", method="GET", status_code="200").inc()
    info = loader.get_model_info()
    return HealthResponse(
        status="healthy",
        service="RailGuard",
        model=info["model_name"],
        alias=info["alias"]
    )

@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    HTTP_REQUESTS_TOTAL.labels(endpoint="/model-info", method="GET", status_code="200").inc()
    info = loader.get_model_info()
    return ModelInfoResponse(**info)

@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)):
    start_time = time.time()
    
    # 1. Image Validation
    if not file.content_type or not file.content_type.startswith("image/"):
        PREDICTION_REQUESTS_TOTAL.labels(status="failed").inc()
        HTTP_REQUESTS_TOTAL.labels(endpoint="/predict", method="POST", status_code="400").inc()
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = file.file.read()
        if len(contents) == 0:
            PREDICTION_REQUESTS_TOTAL.labels(status="failed").inc()
            HTTP_REQUESTS_TOTAL.labels(endpoint="/predict", method="POST", status_code="400").inc()
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
            
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image.thumbnail((640, 640))
    except Exception as e:
        PREDICTION_REQUESTS_TOTAL.labels(status="failed").inc()
        HTTP_REQUESTS_TOTAL.labels(endpoint="/predict", method="POST", status_code="400").inc()
        raise HTTPException(status_code=400, detail="Corrupted image file.")

    # 2. Inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference_start = time.time()
    try:
        results = model(image, conf=CONFIDENCE_THRESHOLD, device=device, imgsz=640)
    except Exception as e:
        import traceback
        traceback.print_exc()
        PREDICTION_REQUESTS_TOTAL.labels(status="failed").inc()
        HTTP_REQUESTS_TOTAL.labels(endpoint="/predict", method="POST", status_code="500").inc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    
    inference_duration = time.time() - inference_start
    INFERENCE_LATENCY.observe(inference_duration)
    
    detections = []
    
    if len(results) > 0:
        result = results[0]
        
        annotated_img = result.plot()
        output_filename = f"pred_{uuid.uuid4().hex[:8]}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        
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
            # Log individual defect class
            DETECTIONS_TOTAL.labels(class_name=class_name).inc()

    # Log metrics
    total_latency = time.time() - start_time
    PREDICTION_LATENCY.observe(total_latency)
    PREDICTION_REQUESTS_TOTAL.labels(status="success").inc()
    HTTP_REQUESTS_TOTAL.labels(endpoint="/predict", method="POST", status_code="200").inc()
    DETECTIONS_PER_REQUEST.observe(len(detections))
    
    info = loader.get_model_info()
    print(f"[INFERENCE LOG] model: {info['model_name']} | version: {info['version']} | alias: {info['alias']} | latency: {total_latency:.4f}s | detections: {len(detections)}")

    return PredictionResponse(detections=detections)

if __name__ == "__main__":
    import uvicorn
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
