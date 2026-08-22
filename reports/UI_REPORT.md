# RailGuard Web UI Report

## 1. UI Architecture
The RailGuard Web UI is built with **Streamlit**. It strictly adheres to a microservices architecture by separating the frontend from the inference logic:
- **Frontend:** Streamlit (`ui/app.py`) manages file uploads, system status polling, JSON parsing, and dynamic bounding box drawing.
- **Backend:** FastAPI (`api/main.py`) handles all YOLOv8 inferences via the MLflow Model Registry.

This ensures the UI is lightweight and the heavy ML operations are handled exclusively by the dedicated API.

## 2. Integration Status
- **FastAPI Integration:** SUCCESS.
- **Health Polling:** Streamlit successfully hits `GET /health` to verify API uptime.
- **Model Metadata:** Streamlit successfully hits `GET /model-info` to display the registered model name, version, and alias in the sidebar.
- **Prediction Pipeline:** Streamlit successfully hits `POST /predict`, uploads the user's image in-memory, receives JSON bounding boxes, and draws them visually for the user without ever touching the local file system.

## 3. Test Results
Integration tests were built using `pytest` and `requests_mock` to verify the Streamlit-facing API integration logic.
- **Upload Valid Image:** Passed.
- **Handle Invalid File (HTTP 400):** Passed.
- **Handle API Unavailable (ConnectionError):** Passed.

## 4. Error Handling
The UI elegantly handles:
- API Unavailability (Greys out detection buttons, shows red status indicators).
- Corrupted or invalid files (Parses FastAPI 400 detail messages and shows Streamlit error banners).
- Empty predictions (Displays a success message: "No defects detected! The track appears clear.").

## 5. Known Limitations
- The UI currently draws bounding boxes synchronously in Python using `Pillow`. For highly complex images with hundreds of bounding boxes, this could introduce minor UI latency, though it is perfectly optimal for standard railway defect scenarios.
- The UI assumes the API is running on `127.0.0.1:8000`. In a production Docker environment, this would need to be injected via environment variables.
