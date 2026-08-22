# RailGuard API Guide

## Overview
This service provides a FastAPI-based REST API for detecting railway track defects using the `RailGuard-Detector` model loaded dynamically from the local MLflow Model Registry.

## How to Start the API
Ensure your Python environment has the necessary packages, then start Uvicorn:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

## Endpoints

### 1. `GET /health`
Returns the status of the service and the model currently loaded.
**Response:**
```json
{
  "status": "healthy",
  "service": "RailGuard",
  "model": "RailGuard-Detector",
  "alias": "candidate"
}
```

### 2. `GET /model-info`
Returns detailed metadata regarding the loaded model from the registry.
**Response:**
```json
{
  "model_name": "RailGuard-Detector",
  "version": "1",
  "alias": "candidate",
  "task": "railway_defect_detection"
}
```

### 3. `POST /predict`
Performs object detection on an uploaded railway image.

**Request Format:** `multipart/form-data` with an image file under the key `file`.
**Example cURL:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_track.jpg"
```

**Response Format:**
```json
{
  "detections": [
    {
      "class_name": "crack",
      "confidence": 0.87,
      "bbox": {
        "x1": 120,
        "y1": 80,
        "x2": 340,
        "y2": 220
      }
    }
  ]
}
```

## Model Source & Configuration
The API does not use hardcoded weight files. It connects to the MLflow database (`sqlite:///mlflow.db`) and dynamically pulls the `RailGuard-Detector` using the alias `candidate`. 

The confidence threshold for predictions is fully configurable in `configs/serve.yaml` (default is 0.25). 
All images submitted for prediction are run through the YOLOv8 engine and annotated copies are saved to `api_outputs/` for debugging.
