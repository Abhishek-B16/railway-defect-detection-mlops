# RailGuard Web UI Guide

## Overview
The Streamlit Web UI provides a clean, user-friendly frontend to interact with the RailGuard defect detection system. It connects exclusively to the FastAPI backend to perform object detection, ensuring the UI remains stateless and decoupled from the model inference engine.

## Quick Start

### 1. Start the FastAPI Backend
Open a terminal and run the inference API:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
- **API URL:** `http://127.0.0.1:8000`

### 2. Start the Streamlit UI
Open a second terminal and run the frontend:
```bash
streamlit run ui/app.py
```
- **UI URL:** `http://localhost:8501` (Streamlit will typically open this automatically in your browser).

## How Image Prediction Works
1. The user uploads an image (`JPG`, `JPEG`, or `PNG`).
2. The user clicks **Detect Defects**.
3. Streamlit sends the image via a `multipart/form-data` POST request to the FastAPI `/predict` endpoint.
4. FastAPI passes the image to the dynamically loaded YOLOv8 `RailGuard-Detector` candidate model.
5. FastAPI returns a JSON response containing bounding boxes and confidence scores.
6. Streamlit parses the JSON, draws the bounding boxes dynamically on the uploaded image in-memory, and displays the final annotated image and defect metrics.

## Troubleshooting
- **API Offline (Red Status):** The sidebar will display "API Offline" if Streamlit cannot reach FastAPI. Ensure `uvicorn` is running and accessible on port 8000.
- **Connection Timed Out:** If the model takes too long to infer (especially on CPU), Streamlit may timeout. You can adjust the `timeout` parameter in `ui/app.py` if needed.
- **Empty Predictions:** If the track is clear, the system will output "No defects detected!".
