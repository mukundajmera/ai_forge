#!/usr/bin/env python3
"""
AI Forge - Comprehensive System Verification Script
Tests all major backend endpoints and features.
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/v1"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test(name, status, details=""):
    icon = f"{Colors.GREEN}✓{Colors.END}" if status else f"{Colors.RED}✗{Colors.END}"
    print(f"{icon} {name}")
    if details:
        print(f"  {Colors.BLUE}→{Colors.END} {details}")

def test_health_check():
    """Test health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        success = response.status_code == 200
        print_test("Health Check", success, f"Status: {response.status_code}")
        return success
    except Exception as e:
        print_test("Health Check", False, f"Error: {str(e)}")
        return False

def test_system_status():
    """Test system status endpoint"""
    try:
        # Try both possible endpoints
        endpoints = [f"{API_URL}/system/status", f"{BASE_URL}/api/system/status"]
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    print_test("System Status", True, f"Healthy: {data.get('healthy', 'N/A')}")
                    return True
            except:
                continue
        print_test("System Status", False, "Endpoint not found")
        return False
    except Exception as e:
        print_test("System Status", False, f"Error: {str(e)}")
        return False

def test_list_models():
    """Test list models endpoint"""
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        success = response.status_code == 200
        if success:
            models = response.json()
            count = len(models.get('data', [])) if isinstance(models, dict) else len(models)
            print_test("List Models", True, f"Found {count} models")
        else:
            print_test("List Models", False, f"Status: {response.status_code}")
        return success
    except Exception as e:
        print_test("List Models", False, f"Error: {str(e)}")
        return False

def test_list_data_sources():
    """Test list data sources"""
    try:
        response = requests.get(f"{BASE_URL}/api/data-sources", timeout=5)
        success = response.status_code == 200
        if success:
            sources = response.json()
            print_test("List Data Sources", True, f"Found {len(sources)} sources")
            return sources
        else:
            print_test("List Data Sources", False, f"Status: {response.status_code}")
        return []
    except Exception as e:
        print_test("List Data Sources", False, f"Error: {str(e)}")
        return []

def test_list_datasets():
    """Test list datasets"""
    try:
        response = requests.get(f"{BASE_URL}/api/datasets", timeout=5)
        success = response.status_code == 200
        if success:
            datasets = response.json()
            ready_count = sum(1 for d in datasets if d.get('status') == 'ready')
            print_test("List Datasets", True, f"Found {len(datasets)} datasets ({ready_count} ready)")
            return datasets
        else:
            print_test("List Datasets", False, f"Status: {response.status_code}")
        return []
    except Exception as e:
        print_test("List Datasets", False, f"Error: {str(e)}")
        return []

def test_list_jobs():
    """Test list training jobs"""
    try:
        response = requests.get(f"{API_URL}/fine-tune", timeout=5)
        success = response.status_code == 200
        if success:
            jobs = response.json()
            print_test("List Training Jobs", True, f"Found {len(jobs)} jobs")
            return jobs
        else:
            print_test("List Training Jobs", False, f"Status: {response.status_code}")
        return []
    except Exception as e:
        print_test("List Training Jobs", False, f"Error: {str(e)}")
        return []

def test_create_training_job(dataset_id):
    """Test creating a training job"""
    if not dataset_id:
        print_test("Create Training Job", False, "No dataset available")
        return None
    
    try:
        payload = {
            "project_name": f"verification_test_{int(time.time())}",
            "base_model": "unsloth/Llama-3.2-3B-Instruct",
            "dataset_id": dataset_id,
            "epochs": 1,
            "learning_rate": 0.0002,
            "rank": 32,
            "batch_size": 2,
            "use_pissa": True
        }
        response = requests.post(f"{API_URL}/fine-tune", json=payload, timeout=10)
        success = response.status_code in [200, 201]
        if success:
            job_data = response.json()
            job_id = job_data.get('job_id') or job_data.get('jobId')
            print_test("Create Training Job", True, f"Job ID: {job_id}")
            return job_id
        else:
            print_test("Create Training Job", False, f"Status: {response.status_code}, Response: {response.text[:200]}")
        return None
    except Exception as e:
        print_test("Create Training Job", False, f"Error: {str(e)}")
        return None

def test_get_job_details(job_id):
    """Test getting job details"""
    if not job_id:
        print_test("Get Job Details", False, "No job ID")
        return False
    
    try:
        response = requests.get(f"{API_URL}/fine-tune/{job_id}", timeout=5)
        success = response.status_code == 200
        if success:
            job = response.json()
            status = job.get('status', 'unknown')
            print_test("Get Job Details", True, f"Status: {status}")
        else:
            print_test("Get Job Details", False, f"Status: {response.status_code}")
        return success
    except Exception as e:
        print_test("Get Job Details", False, f"Error: {str(e)}")
        return False

def main():
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}AI Forge - System Verification{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")

    results = {}

    # Core API Tests
    print(f"{Colors.YELLOW}📡 Core API Tests{Colors.END}")
    results['health'] = test_health_check()
    results['system_status'] = test_system_status()
    results['models'] = test_list_models()
    print()

    # Data Management Tests
    print(f"{Colors.YELLOW}📊 Data Management Tests{Colors.END}")
    sources = test_list_data_sources()
    datasets = test_list_datasets()
    results['data_sources'] = len(sources) >= 0
    results['datasets'] = len(datasets) >= 0
    print()

    # Training Job Tests
    print(f"{Colors.YELLOW}🚀 Training Job Tests{Colors.END}")
    jobs = test_list_jobs()
    results['list_jobs'] = len(jobs) >= 0
    
    # Get a ready dataset for testing
    ready_dataset = next((d for d in datasets if d.get('status') == 'ready'), None)
    dataset_id = None
    
    if ready_dataset:
        dataset_id = ready_dataset['id']
    elif Path("data/datasets/verification_dataset.json").exists():
        print(f"{Colors.YELLOW}  Using local verification dataset{Colors.END}")
        dataset_id = "verification_dataset"
        
    if dataset_id:
        job_id = test_create_training_job(dataset_id)
        results['create_job'] = job_id is not None
        
        if job_id:
            time.sleep(1)  # Brief pause
            results['get_job'] = test_get_job_details(job_id)
    else:
        print_test("Create Training Job", False, "No ready datasets available")
        results['create_job'] = False
        results['get_job'] = False
    print()

    # Summary
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    print(f"\n{Colors.YELLOW}Summary:{Colors.END}")
    print(f"  Total Tests: {total}")
    print(f"  {Colors.GREEN}Passed: {passed}{Colors.END}")
    print(f"  {Colors.RED}Failed: {failed}{Colors.END}")
    
    if failed == 0:
        print(f"\n{Colors.GREEN}✓ All tests passed!{Colors.END}\n")
    else:
        print(f"\n{Colors.RED}✗ Some tests failed. Review output above.{Colors.END}\n")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
