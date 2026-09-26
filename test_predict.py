import requests
import sys
import json

url = "https://railguard-api-tr8j.onrender.com/predict"
image_path = "railway defect dataset.v9i.yolov8/train/images/1_MOV_20201221091849_3610_JPEG.rf.05f0254a3ce149b597e36b880a829ffe.jpg"

try:
    with open(image_path, "rb") as f:
        files = {"file": (image_path.split("/")[-1], f, "image/jpeg")}
        print(f"Sending request to {url}...")
        response = requests.post(url, files=files, timeout=60)
        
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    print(f"Response text (first 500 chars): {response.text[:500]}")
    
    try:
        data = response.json()
        print(f"JSON Response: {json.dumps(data, indent=2)}")
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        
except Exception as e:
    print(f"Error: {e}")
