# RailGuard Deployment Plan

## 1. Project Goal
Deploy the RailGuard YOLOv8 defect detection system to a publicly accessible, free-tier cloud platform for a college MLOps project demonstration.

## 2. Current Architecture
- **Frontend**: Streamlit Web UI (`railguard-ui`)
- **Backend**: FastAPI Server (`railguard-api`)
- **Model Storage**: Local SQLite MLflow Database (`mlflow.db`) and artifact store (`mlruns/`) copied directly into the Docker image.
- **Inference**: YOLOv8 Nano running via Ultralytics and PyTorch.
- **Orchestration**: Docker Compose.

## 3. Recommended Cloud Architecture
**Platform**: Render.com (Free Tier) or AWS EC2 t2.micro (AWS Free Tier).
**Recommendation**: **AWS EC2 (t2.micro)** is recommended for this specific architecture.

### Why AWS EC2?
- Our architecture relies on **Docker Compose** with two distinct services (`api` and `ui`) communicating over a bridge network.
- PaaS free tiers (like Render, Heroku) often restrict deployments to a single exposed web port per service, and their 512MB RAM limits are typically too small to load PyTorch and YOLO into memory without crashing (OOM errors).
- AWS EC2 t2.micro provides a full virtual machine (1GB RAM + configurable Swap Space). We can install Docker Desktop/Engine and seamlessly run our exact `docker-compose.yml` with zero architectural changes.

### Where the Model Lives
For a production enterprise system, `best.pt` would live in an AWS S3 bucket, and MLflow would be hosted on a separate tracking server. 
**However, for this MLOps college project**, the current approach of "baking" the `mlruns` directory and `mlflow.db` directly into the `railguard-api` Docker image is highly recommended. It keeps the deployment architecture simple, free, and robust. The API container queries its internal read-only MLflow registry on startup to load the model.

### Compute Requirements
- **GPU**: Not required. YOLOv8 Nano is highly optimized. CPU inference for a single image upload will take approximately 100-300ms, which is perfectly acceptable for a web UI demo.
- **RAM**: PyTorch and FastAPI will consume roughly 500-700MB. Streamlit will consume ~200MB. We will configure a 2GB Swap File on the EC2 instance to prevent Out-Of-Memory (OOM) crashes.

## 4. Required Changes Before Deployment (COMPLETED)
The codebase has been fully updated for cloud deployment:
1. **Removed `api_outputs` volume mount**: The local host mapping in `docker-compose.yml` has been removed. The container now natively handles output saving internally.
2. **Hardened FastAPI `content_type`**: The prediction endpoint is now robust against clients sending `None` or missing content types.
3. **MLflow Pathing Verified**: The SQLite absolute path rewriting is verified to work cross-platform.

## 5. Deployment Flow
1. Provision a free-tier Ubuntu t2.micro instance on AWS EC2.
2. Configure Security Group to allow inbound traffic on TCP 8501 and 22 (SSH).
3. SSH into the instance, allocate 2GB of swap memory.
4. Install Docker and Docker Compose.
5. `git clone` the repository onto the instance.
6. Run `docker compose up -d --build`.
7. Access the public URL: `http://<EC2-PUBLIC-IP>:8501`.

## 6. Risks & Limitations
- **Cold Starts**: Not an issue on EC2 (runs 24/7), but if deployed on a serverless platform (Render/Heroku), the API would sleep after 15 minutes of inactivity, causing a 60-second delay on the next request.
- **Memory Limits**: 1GB of RAM is tight for PyTorch. Swap space is strictly required.
