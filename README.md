# 🚂 RailGuard: Railway Defect Detection MLOps

An end-to-end Machine Learning Operations (MLOps) pipeline for detecting structural defects (cracks, flaking, spalling, missing bolts, etc.) on railway tracks.

This project demonstrates a production-ready, containerized architecture that handles data versioning, model tracking, dynamic model loading, and serving via a decoupled frontend and backend.

## 🏗️ Architecture

The system is built on a modern MLOps stack, designed to seamlessly transition models from training into cloud deployment.

1. **DVC (Data Version Control):** Tracks the dataset versions (`dataset_clean`, `dataset_detection`).
2. **YOLOv8 (Ultralytics):** The core computer vision detection model.
3. **MLflow & DagsHub:** Acts as the Model Registry. Training metrics and the `best.pt` weights are pushed here.
4. **FastAPI (Backend):** A lightweight API that automatically connects to DagsHub on startup, downloads the model marked with the `candidate` alias, and serves predictions.
5. **Streamlit (Frontend):** A purely presentational UI that communicates with the FastAPI backend over HTTP.
6. **Docker & Render:** The entire system is containerized (`Dockerfile.api`, `Dockerfile.ui`) and orchestrated using Docker Compose for local testing, and deployed to Render for production.

## 🛠️ Tech Stack
* **Machine Learning:** PyTorch, Ultralytics (YOLOv8)
* **MLOps:** MLflow, DVC, DagsHub
* **Backend:** FastAPI, Uvicorn
* **Frontend:** Streamlit
* **Deployment:** Docker, Docker Compose, Render

## 🚀 How to Run Locally

1. **Clone the repository and navigate into it:**
   ```bash
   git clone <your-repo-url>
   cd railway-defect-mlops
   ```

2. **Run using Docker Compose:**
   ```bash
   docker compose up --build
   ```

3. **Access the Application:**
   * UI (Streamlit): `http://localhost:8501`
   * API (FastAPI): `http://localhost:8000`
   * API Interactive Docs: `http://localhost:8000/docs`

## ⚠️ Known Limitations & Future Work

As this project is currently deployed on a **Free Tier** cloud environment (Render), specific architectural compromises were made to prevent Out-Of-Memory (OOM) crashes:

* **Inference Resolution:** Render's Free Tier is limited to 512MB RAM. Standard YOLOv8 inference at `640x640` causes memory spikes that crash the server. To solve this, the inference resolution is strictly forced down to `320x320`, and PyTorch memory is pre-allocated (warmed up) during API startup.
* **Model Accuracy (mAP50 ~ 55%):** Because of the required `320x320` downscaling and the limitations of the current dataset size, the model occasionally struggles to detect fine-grained cracks on out-of-distribution, high-resolution real-world images (e.g., random images from Google). 

**Future Work:**
* **Upgrade Cloud Hardware:** Move the FastAPI service to a machine with >1GB RAM to support native `640x640` inference.
* **Data Flywheel:** Implement an Active Learning loop by taking failing out-of-distribution images, annotating them, and tracking them via DVC to continuously improve the model's generalization capabilities.