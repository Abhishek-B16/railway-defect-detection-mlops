# RailGuard Docker Guide

## Prerequisites
- Docker Desktop (or Docker Engine) installed and running.
- Docker Compose installed (comes bundled with Docker Desktop).

## Build Command
To build the Docker images for both the API and Streamlit UI, run the following command from the root directory of the project:
```bash
docker compose build
```

## Start Command
To start the entire RailGuard system in detached mode (background):
```bash
docker compose up -d
```

## Stop Command
To stop and remove the containers:
```bash
docker compose down
```

## Service URLs
Once the containers are up and running, you can access the services at:
- **Streamlit Web UI:** `http://localhost:8501`
- **FastAPI Inference Server:** `http://localhost:8000`
- **API Health Check:** `http://localhost:8000/health`
- **Model Info Endpoint:** `http://localhost:8000/model-info`

## Architecture & Integration
Docker Compose creates an internal network where the `ui` service communicates with the `api` service using the hostname `api`. 
The `API_URL` environment variable is explicitly passed in `docker-compose.yml` to instruct Streamlit to send inference requests to `http://api:8000`.

The `mlflow.db` SQLite database and the `mlruns/` directories are mapped into the `api` container so that the inference service can natively query the Model Registry and dynamically pull the `RailGuard-Detector@candidate` model directly, without hardcoding `.pt` file paths.

## Troubleshooting
- **Port Conflicts:** If ports 8000 or 8501 are already in use on your host machine, you can change the mapping in `docker-compose.yml` (e.g., `"8080:8000"`).
- **Model Not Found in Docker:** Ensure that `mlflow.db` and the `mlruns/` directory are successfully copied into the API container (this is handled by `Dockerfile.api`).
- **Cannot Reach API from UI:** Ensure you are not hardcoding `localhost` in the `app.py` when running inside Docker. Streamlit must use the Docker bridge network hostname (`api`).
