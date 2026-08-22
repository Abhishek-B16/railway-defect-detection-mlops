# RailGuard Model Serving Report

## 1. Architecture Overview
The inference service acts as a robust model serving layer.
- **Client** submits images to the FastAPI `/predict` endpoint.
- **Model Loader** interfaces directly with the MLflow tracking SQLite database and Model Registry.
- **YOLOv8** executes predictions.
- **FastAPI** packages bounding boxes, class names, and confidence scores into a JSON schema and serves it back to the client.

## 2. API Endpoints Created
- `GET /health` : Verifies service and model load status.
- `GET /model-info` : Reads MLflow tags and exposes model version/alias data.
- `POST /predict` : Handles `multipart/form-data` image uploads, runs validation, executes object detection, and logs latency and predictions.

## 3. Model Registry Integration
The serving service successfully dynamically pulls the `RailGuard-Detector` model via the MLflow Model Registry alias `candidate`. No `.pt` weight files are hardcoded in the codebase, proving that the deployment natively interfaces with our MLOps lifecycle tracking.

## 4. Test Results
Comprehensive tests were built using `pytest` and FastAPI's `TestClient`.
- `/health` and `/model-info` return valid `200` codes.
- `POST /predict` validates empty files (`422 Unprocessable Entity`).
- `POST /predict` validates corrupted images (`400 Bad Request`).
- `POST /predict` validates non-image text files (`400 Bad Request`).
- `POST /predict` correctly serves bounding boxes and annotations for valid images.

## 5. Output Management
Annotated visual predictions are saved to `api_outputs/` so that developers can visually audit the model's confidence logic. The directory is intentionally added to `.gitignore` to prevent bloating the Git repository with diagnostic outputs.

## 6. Known Limitations
- MLOps logging currently outputs structured strings to stdout (`[INFERENCE LOG] ... latency: 0.1s`). While suitable for this baseline, a real production system should pipe these to structured log aggregators or push metrics back into MLflow as asynchronous events.
- Because it's using standard Python `TestClient`, concurrency wasn't deeply tested here. The API handles concurrent async uploads, but the YOLO inference is currently synchronous, meaning high-traffic blocking may occur without standard async worker queues.
