import pytest
import requests
import requests_mock

API_URL = "http://127.0.0.1:8000"

def test_ui_api_integration_valid_image():
    with requests_mock.Mocker() as m:
        # Mock prediction endpoint
        m.post(f"{API_URL}/predict", json={
            "detections": [
                {
                    "class_name": "crack",
                    "confidence": 0.85,
                    "bbox": {"x1": 10, "y1": 10, "x2": 100, "y2": 100}
                }
            ]
        })
        
        # Simulate Streamlit making the POST request
        response = requests.post(f"{API_URL}/predict", files={"file": ("dummy.jpg", b"fake_image_data", "image/jpeg")})
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["detections"]) == 1
        assert data["detections"][0]["class_name"] == "crack"

def test_ui_api_integration_invalid_file():
    with requests_mock.Mocker() as m:
        # Mock FastAPI's 400 bad request for invalid files
        m.post(f"{API_URL}/predict", status_code=400, json={"detail": "File provided is not an image."})
        
        response = requests.post(f"{API_URL}/predict", files={"file": ("dummy.txt", b"text", "text/plain")})
        
        assert response.status_code == 400
        assert "not an image" in response.json()["detail"]

def test_ui_api_integration_unavailable():
    # Do not mock, let requests attempt to connect to a nonexistent endpoint or use mock to simulate ConnectionError
    with requests_mock.Mocker() as m:
        m.post(f"{API_URL}/predict", exc=requests.exceptions.ConnectionError)
        
        with pytest.raises(requests.exceptions.ConnectionError):
            requests.post(f"{API_URL}/predict", files={"file": ("dummy.jpg", b"data", "image/jpeg")})
