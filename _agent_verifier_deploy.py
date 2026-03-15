import requests
import sys

def verify_deploy():
    job_id = "job_test_cpu_job_20260316_022933"
    url = f"http://localhost:8000/models/{job_id}/deploy"
    
    print(f"Triggering deployment for job {job_id} at {url}...")
    try:
        response = requests.post(url, json={"model_name": f"deployed-{job_id}"})
        
        if response.status_code == 200:
            print("Deployment succeeded!")
            print(response.json())
            sys.exit(0)
        else:
            print(f"Deployment failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during deployment request: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_deploy()
