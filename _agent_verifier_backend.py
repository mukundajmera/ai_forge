import sys
import time
import requests

def test_models_api():
    print("Testing /v1/models API...")
    try:
        resp = requests.get("http://localhost:8000/v1/models", timeout=5)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            print("FAIL: No models returned from /v1/models")
            return False
        print(f"PASS: Models returned: {[m['id'] for m in data]}")
        return True
    except Exception as e:
        print(f"FAIL: /v1/models raised error: {e}")
        # Might be useful to print response content if available
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
        return False

def test_cpu_training():
    print("Testing CPU training request...")
    try:
        payload = {
            "project_name": "test_cpu_job",
            "base_model": "unsloth/Qwen2.5-0.5B-Instruct", 
            "epochs": 1
        }
        resp = requests.post("http://localhost:8000/v1/fine-tune", json=payload, timeout=5)
        if resp.status_code != 200:
            print(f"FAIL: /v1/fine-tune returned {resp.status_code}: {resp.text}")
            return False
        
        job_id = resp.json()["job_id"]
        print(f"Job started: {job_id}. Waiting for status...")
        
        for _ in range(60):
            status_resp = requests.get(f"http://localhost:8000/v1/fine-tune/{job_id}", timeout=5)
            status_data = status_resp.json()
            status = status_data["status"]
            print(f"Current status: {status}")
            if status == "failed":
                print(f"FAIL: Training failed: {status_data.get('error')}")
                return False
            if status == "completed":
                print("PASS: Training completed")
                return True
            time.sleep(2)
        print("FAIL: Training timed out (stuck in queued/training)")
        return False
    except Exception as e:
        print(f"FAIL: CPU training test raised error: {e}")
        return False

if __name__ == "__main__":
    success = True
    if not test_models_api():
        success = False
    
    if not test_cpu_training():
        success = False
        
    if not success:
        sys.exit(1)
    print("ALL PASS")
    sys.exit(0)
