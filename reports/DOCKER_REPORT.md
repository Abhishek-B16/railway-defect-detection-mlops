# RailGuard Dockerization Report

## 1. Container Architecture
The RailGuard system has been successfully configured for containerization using Docker Compose. The architecture consists of two microservices that operate on a unified internal Docker bridge network:

1. **`railguard-api` (FastAPI Server)**
   - Exposes Port: `8000`
   - Image built via: `Dockerfile.api`
   - Responsibility: Hosts the YOLOv8 inference engine. It is configured to natively read the `mlflow.db` registry database and access the `mlruns/` artifact store to dynamically pull the `RailGuard-Detector@candidate` model into memory on startup.

2. **`railguard-ui` (Streamlit Frontend)**
   - Exposes Port: `8501`
   - Image built via: `Dockerfile.ui`
   - Responsibility: Provides the interactive user interface. It communicates exclusively with the backend via the `API_URL` environment variable (`http://api:8000`).

## 2. Best Practices Implemented
- **`.dockerignore` Created:** Excludes massive datasets (`dataset_clean/`, `dataset_detection/`), cached DVC data, Python `__pycache__`, and `.git` objects. This keeps the image sizes dramatically smaller and builds significantly faster.
- **Microservice Decoupling:** The UI is completely unaware of the YOLO library or MLflow registry. It acts as a pure presentation layer.
- **Environment Variables:** The UI natively reads the `API_URL` from the OS environment injected by `docker-compose.yml`, avoiding hardcoded `localhost` issues within container networks.
- **Volume Mapping:** Created a local `api_outputs/` volume map so that annotated images generated inside the container can be inspected from the host machine for debugging purposes.

## 3. Actual Build & Runtime Results (Phase 10 Final Verification)
During the final verification phase, the following actions were successfully executed:
- **Dependency Resolution**: Created an explicit `requirements.txt` for the API container to pin `fastapi`, `uvicorn`, `ultralytics`, `mlflow`, etc., replacing the missing dependency file.
- **Docker Build**: `docker compose build` executed flawlessly, building both `railguard-api` (using Debian `libgl1`) and `railguard-ui` successfully.
- **Service Deployment**: `docker compose up -d` started both containers on the bridge network.
- **End-to-End Prediction**: 
  - `http://localhost:8000/health` reports status `healthy` with the resolved model metadata `RailGuard-Detector@candidate`.
  - An image uploaded via POST `/predict` correctly parsed through FastAPI, invoked the MLflow Registry, and yielded bounding box coordinates for `bolts` at `0.92` confidence.
  - `http://localhost:8501` is serving the Streamlit frontend.

## 4. Problems Encountered & Diagnosis
- **Missing Requirements**: A `requirements.txt` file was not present in the root directory. To resolve this, a fully-pinned backend dependency file was created instead of mocking an incomplete one.
- **Debian Trixie Deprecation**: The `python:3.10-slim` base image upgraded to Debian Trixie which deprecated `libgl1-mesa-glx`. This was diagnosed and fixed by swapping the dependency to `libgl1`.

## 5. Conclusion
Phase 10 Dockerization is definitively complete and verified. The complete prediction pipeline executes fully containerized without runtime pathing issues.
