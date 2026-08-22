# RailGuard Deployment Readiness Report

## Executive Summary
The RailGuard system has been thoroughly analyzed for cloud deployment readiness. The architecture successfully decouples the frontend and backend, leverages a localized MLflow registry, and uses containerization. The system is **READY** for deployment with minor configuration adjustments.

## 1. Component Readiness

### Docker Configuration
- **Status**: Excellent.
- **Analysis**: `.dockerignore` correctly prevents the 4,628-image training dataset and DVC caches from inflating the image size. The build context is lean.

### MLflow Registry Architecture
- **Status**: Ready for "Baked" Deployment.
- **Analysis**: The current system copies `mlflow.db` and `mlruns/` into the `Dockerfile.api` image. While an enterprise system would host MLflow on a separate cloud server and use AWS S3 for artifacts, this "baked-in" read-only registry is the absolute best approach for a free-tier college project. It avoids the immense cost and complexity of a distributed MLflow setup while still demonstrating the registry concept.
- **Pathing Fixes**: The absolute Windows paths embedded in the SQLite database were successfully patched in Phase 10 (`model_loader.py`), meaning the Linux container resolves paths safely.

### Inference Backend (FastAPI)
- **Status**: Ready.
- **Analysis**: Dependencies are properly pinned in `requirements.txt`. The API relies on CPU inference. Since YOLOv8 Nano (`yolov8n`) is lightweight, CPU inference is more than sufficient for HTTP-based image upload processing.

### Frontend (Streamlit)
- **Status**: Ready.
- **Analysis**: The UI communicates dynamically via the `API_URL` environment variable. It is completely unaware of the host environment, making it highly portable.

## 2. Resource Estimation
- **API Service**: ~600MB RAM (PyTorch + Ultralytics overhead).
- **UI Service**: ~200MB RAM.
- **Storage**: ~2GB (Docker images + OS overhead).
- **Total RAM**: ~800MB - 1GB.
- *Note*: Deploying to a 512MB free-tier PaaS (like Render or Heroku) will likely result in an Out-Of-Memory (OOM) error. A Virtual Machine with 1GB RAM and a configured Swap file (e.g., AWS EC2 t2.micro) is strongly recommended.

## 3. Necessary Pre-Deployment Actions
1. **Remove Volume Mounts**: Remove the local `./api_outputs` volume mapping in `docker-compose.yml` as it is not needed in a cloud environment.
2. **Swap Configuration**: The deployment script/guide must include instructions to enable Linux Swap memory to handle PyTorch load spikes.

## Verdict
The project demonstrates excellent MLOps practices. It is modular, lightweight, and deployment-ready.
