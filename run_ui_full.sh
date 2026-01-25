#!/bin/bash
echo "Stopping existing processes..."
pkill -f uvicorn || true
pkill -f "next dev" || true

echo "Activating venv and installing requirements..."
# source .venv/bin/activate
./.venv/bin/pip install -r requirements.txt

echo "Starting Backend..."
./.venv/bin/python -m uvicorn conductor.service:app --reload --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

echo "Starting Frontend..."
cd frontend
npm run dev -- -p 3000 > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

echo "Application stack running."
