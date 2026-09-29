from fastapi.testclient import TestClient
from api.main import app
import sys

def test_all():
    client = TestClient(app)
    
    print("Testing /health...")
    r = client.get("/health")
    print(r.status_code, r.json())

    print("Testing /model-info...")
    r = client.get("/model-info")
    print(r.status_code, r.json())

    print("Testing /metrics...")
    r = client.get("/metrics")
    print(r.status_code)
    metrics_text = r.text
    if 'railguard_api_health' in metrics_text:
        print("SUCCESS: Found railguard_api_health metric.")
    else:
        print("FAILED: Could not find custom metrics.")
        sys.exit(1)
        
    if 'railguard_http_requests_total' in metrics_text:
        print("SUCCESS: Found HTTP requests metric.")
    else:
        print("FAILED: Could not find HTTP metric.")
        sys.exit(1)
        
    print("ALL TESTS PASSED.")

if __name__ == '__main__':
    test_all()
