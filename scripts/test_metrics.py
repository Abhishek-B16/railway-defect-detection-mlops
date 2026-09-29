import requests
import json
import time

def test_api():
    print("Testing /health...")
    r = requests.get("http://127.0.0.1:8000/health")
    print(r.status_code, r.json())

    print("Testing /model-info...")
    r = requests.get("http://127.0.0.1:8000/model-info")
    print(r.status_code, r.json())

    print("Testing /metrics...")
    r = requests.get("http://127.0.0.1:8000/metrics")
    print(r.status_code)
    metrics_text = r.text
    if 'railguard_api_health' in metrics_text:
        print("SUCCESS: Found railguard_api_health metric.")
    else:
        print("FAILED: Could not find custom metrics.")

if __name__ == '__main__':
    test_api()
