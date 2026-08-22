import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io

from api.main import app

@pytest.fixture(scope="module")
def client():
    # Setup test files
    os.makedirs("tests/data", exist_ok=True)
    
    # 1. Valid image (black square, likely no detections)
    img = Image.new('RGB', (640, 640), color='black')
    valid_path = "tests/data/valid.jpg"
    img.save(valid_path)
    
    # 2. Invalid text file
    invalid_path = "tests/data/invalid.txt"
    with open(invalid_path, "w") as f:
        f.write("This is a text file.")
        
    # 3. Corrupted image
    corrupt_path = "tests/data/corrupted.jpg"
    with open(corrupt_path, "wb") as f:
        f.write(b"not an image data at all")

    # Yield the TestClient as a context manager so startup events run
    with TestClient(app) as client:
        yield client

    # Cleanup
    if os.path.exists(valid_path): os.remove(valid_path)
    if os.path.exists(invalid_path): os.remove(invalid_path)
    if os.path.exists(corrupt_path): os.remove(corrupt_path)

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "RailGuard"
    assert "model" in data
    assert "alias" in data

def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "RailGuard-Detector"
    assert data["alias"] == "candidate"
    assert data["task"] == "railway_defect_detection"

def test_predict_valid_image(client):
    with open("tests/data/valid.jpg", "rb") as f:
        response = client.post("/predict", files={"file": ("valid.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    # Empty image should have 0 detections
    assert isinstance(data["detections"], list)

def test_predict_invalid_file(client):
    with open("tests/data/invalid.txt", "rb") as f:
        response = client.post("/predict", files={"file": ("invalid.txt", f, "text/plain")})
    assert response.status_code == 400
    assert "not an image" in response.json()["detail"]

def test_predict_corrupted_image(client):
    with open("tests/data/corrupted.jpg", "rb") as f:
        response = client.post("/predict", files={"file": ("corrupted.jpg", f, "image/jpeg")})
    assert response.status_code == 400
    assert "Corrupted" in response.json()["detail"]

def test_predict_empty_upload(client):
    # Missing file field entirely
    response = client.post("/predict")
    assert response.status_code == 422 # FastAPI validation error for missing field
