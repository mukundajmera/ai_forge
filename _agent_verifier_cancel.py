import time
import requests
import os
import psutil
import json

print("\n--- Testing Job Cancellation & Process Cleanup ---\n")

payload = {
    "project_name": "test_cancel",
    "base_model": "unsloth/Llama-3.2-1B-Instruct",
    "dataset": "dummy",
    "epochs": 1,
    "batch_size": 2,
    "learning_rate": 0.0002,
    "rank": 16,
    "device": "cpu"
}

def get_ai_forge_processes():
    processes = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = " ".join(p.info.get('cmdline') or [])
            if "training_worker.py" in cmd or "forge.py" in cmd:
                processes.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return processes

# 1. Start Job
print("Submitting /v1/fine-tune request...")
res = requests.post("http://localhost:8000/v1/fine-tune", json=payload)
if res.status_code != 200:
    print(f"Failed to start job: {res.text}")
    exit(1)

job_id = res.json().get("job_id")
print(f"Job started: {job_id}")

# 2. Give the worker some time to spawn process tree (Datloader workers, etc.)
print("Waiting 5 seconds for worker process to fully spawn...")
time.sleep(5)

# 3. Check processes before cancellation
running_procs = get_ai_forge_processes()
print(f"Found {len(running_procs)} AI Forge training worker processes running.")
for p in running_procs:
    print(f"  PID: {p.info['pid']} - CMD: {' '.join(p.info['cmdline'])}")

# 4. Cancel Job
print(f"\nCancelling job via DELETE /v1/fine-tune/{job_id}...")
cancel_res = requests.delete(f"http://localhost:8000/v1/fine-tune/{job_id}")
print(f"Cancel Response: {cancel_res.status_code} - {cancel_res.text}")

# 5. Wait for OS to clean up
print("Waiting 3 seconds for OS cleanup...")
time.sleep(3)

# 6. Verify Process Termination
after_procs = get_ai_forge_processes()
if after_procs:
    print(f"FAIL: Found {len(after_procs)} zombie/orphaned processes!")
    for p in after_procs:
        try:
            print(f"  ZOMBIE PID: {p.pid} - Status: {p.status()}")
        except psutil.NoSuchProcess:
            pass
    exit(1)
else:
    print("PASS: All worker processes successfully killed. Clean cancellation verified!")
    exit(0)
