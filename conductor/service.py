"""FastAPI Service - REST API for AI Forge.

This module provides a production-ready FastAPI service with
OpenAI-compatible endpoints for fine-tuning and inference.

Features:
    - Fine-tuning job management
    - Model listing and management
    - Chat completions (OpenAI-compatible)
    - Health checks and monitoring

Example:
    >>> uvicorn ai_forge.conductor.service:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Pydantic Models
# --------------------------------------------------------------------------

class FineTuneRequest(BaseModel):
    """Request to start a fine-tuning job.
    
    Attributes:
        project_name: Name of the project.
        base_model: Base model to fine-tune.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        rank: LoRA/PiSSA rank.
        use_pissa: Whether to use PiSSA initialization.
    """
    
    project_name: str
    base_model: str = "unsloth/Llama-3.2-3B-Instruct"
    epochs: int = 3
    learning_rate: float = 2e-4
    rank: int = 64
    use_pissa: bool = True


class FineTuneResponse(BaseModel):
    """Response from fine-tuning job creation.
    
    Attributes:
        job_id: Unique job identifier.
        status: Job status.
        message: Status message.
    """
    
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    """Status of a fine-tuning job.
    
    Attributes:
        job_id: Job identifier.
        status: Current status.
        progress: Progress percentage (0-100).
        current_epoch: Current training epoch.
        current_step: Current training step.
        loss: Current loss value.
        error: Error message if failed.
        created_at: Job creation time.
        updated_at: Last update time.
    """
    
    job_id: str
    status: str  # queued, training, completed, failed
    progress: float = 0.0
    current_epoch: Optional[int] = None
    current_step: Optional[int] = None
    loss: Optional[float] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class ChatMessage(BaseModel):
    """Chat message.
    
    Attributes:
        role: Message role.
        content: Message content.
    """
    
    role: str
    content: str


class ChatRequest(BaseModel):
    """OpenAI-compatible chat request.
    
    Attributes:
        model: Model name.
        messages: List of messages.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        stream: Whether to stream response.
    """
    
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 256
    stream: bool = False


class ChatChoice(BaseModel):
    """Chat response choice."""
    
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    """OpenAI-compatible chat response.
    
    Attributes:
        id: Response ID.
        object: Object type.
        created: Timestamp.
        model: Model used.
        choices: Response choices.
    """
    
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]


class ModelInfo(BaseModel):
    """Model information.
    
    Attributes:
        id: Model identifier.
        object: Object type.
        owned_by: Owner.
        created: Creation timestamp.
    """
    
    id: str
    object: str = "model"
    owned_by: str = "ai_forge"
    created: int


class HealthResponse(BaseModel):
    """Health check response.
    
    Attributes:
        status: Health status.
        version: Service version.
        ollama_connected: Whether Ollama is connected.
    """
    
    status: str
    version: str
    ollama_connected: bool


# --------------------------------------------------------------------------
# Application State
# --------------------------------------------------------------------------

class AppState:
    """Application state container."""
    
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.ollama_manager: Optional[object] = None
        self.job_queue: Optional[object] = None


state = AppState()


# --------------------------------------------------------------------------
# Lifespan Management
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting AI Forge service...")
    
    # Initialize components
    try:
        from ai_forge.conductor.ollama_manager import OllamaManager
        from ai_forge.conductor.job_queue import JobQueue
        
        state.ollama_manager = OllamaManager()
        state.job_queue = JobQueue()
        
        logger.info("Service initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize some components: {e}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down AI Forge service...")


# --------------------------------------------------------------------------
# Application Factory
# --------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application.
    """
    application = FastAPI(
        title="AI Forge",
        description="Local LLM Fine-Tuning Service",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return application


# Create default application
app = create_app()


# --------------------------------------------------------------------------
# Health & Status Endpoints
# --------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Check service health."""
    ollama_connected = False
    
    if state.ollama_manager:
        try:
            ollama_connected = await state.ollama_manager.health_check()
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        ollama_connected=ollama_connected,
    )


@app.get("/", tags=["Health"])
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "service": "AI Forge",
        "version": "1.0.0",
        "docs": "/docs",
    }


# --------------------------------------------------------------------------
# Fine-Tuning Endpoints
# --------------------------------------------------------------------------

@app.post("/v1/fine-tune", response_model=FineTuneResponse, tags=["Fine-Tuning"])
async def start_fine_tune(
    request: FineTuneRequest,
    background_tasks: BackgroundTasks,
    data_file: UploadFile = File(...),
) -> FineTuneResponse:
    """Start a fine-tuning job.
    
    Upload training data and start a new fine-tuning job.
    
    Args:
        request: Fine-tuning configuration.
        background_tasks: FastAPI background tasks.
        data_file: Training data file (JSON format).
        
    Returns:
        Job creation response with job ID.
    """
    # Generate job ID
    job_id = f"job_{request.project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Save uploaded file
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    data_path = data_dir / f"{job_id}_data.json"
    
    content = await data_file.read()
    data_path.write_bytes(content)
    
    # Create job record
    now = datetime.now().isoformat()
    state.jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "config": request.model_dump(),
        "data_path": str(data_path),
        "created_at": now,
        "updated_at": now,
    }
    
    # Queue background task
    background_tasks.add_task(execute_fine_tune, job_id)
    
    return FineTuneResponse(
        job_id=job_id,
        status="queued",
        message=f"Fine-tuning job queued. Check status at /v1/fine-tune/{job_id}",
    )


@app.get("/v1/fine-tune/{job_id}", response_model=JobStatus, tags=["Fine-Tuning"])
async def get_job_status(job_id: str) -> JobStatus:
    """Get fine-tuning job status.
    
    Args:
        job_id: Job identifier.
        
    Returns:
        Current job status.
    """
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = state.jobs[job_id]
    return JobStatus(**{
        k: v for k, v in job.items()
        if k in JobStatus.model_fields
    })


@app.get("/v1/fine-tune", tags=["Fine-Tuning"])
async def list_jobs() -> list[JobStatus]:
    """List all fine-tuning jobs.
    
    Returns:
        List of job statuses.
    """
    return [
        JobStatus(**{k: v for k, v in job.items() if k in JobStatus.model_fields})
        for job in state.jobs.values()
    ]


@app.delete("/v1/fine-tune/{job_id}", tags=["Fine-Tuning"])
async def cancel_job(job_id: str) -> dict[str, str]:
    """Cancel a fine-tuning job.
    
    Args:
        job_id: Job identifier.
        
    Returns:
        Cancellation confirmation.
    """
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = state.jobs[job_id]
    if job["status"] in ("completed", "failed"):
        raise HTTPException(status_code=400, detail="Job already finished")
    
    job["status"] = "cancelled"
    job["updated_at"] = datetime.now().isoformat()
    
    return {"message": f"Job {job_id} cancelled"}


# --------------------------------------------------------------------------
# Chat Endpoints (OpenAI-Compatible)
# --------------------------------------------------------------------------

@app.post("/v1/chat/completions", response_model=ChatResponse, tags=["Chat"])
async def chat_completions(request: ChatRequest) -> ChatResponse:
    """OpenAI-compatible chat completions endpoint.
    
    Args:
        request: Chat request with messages.
        
    Returns:
        Chat completion response.
    """
    if state.ollama_manager is None:
        raise HTTPException(status_code=503, detail="Ollama not available")
    
    try:
        response = await state.ollama_manager.chat(
            model=request.model,
            messages=[m.model_dump() for m in request.messages],
            temperature=request.temperature,
        )
        
        return ChatResponse(
            id=f"chatcmpl-{datetime.now().timestamp()}",
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=response),
                )
            ],
        )
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------
# Model Management Endpoints
# --------------------------------------------------------------------------

@app.get("/v1/models", tags=["Models"])
async def list_models() -> dict[str, list[ModelInfo]]:
    """List available models.
    
    Returns:
        List of available models.
    """
    models: list[ModelInfo] = []
    
    if state.ollama_manager:
        try:
            ollama_models = await state.ollama_manager.list_models()
            for model in ollama_models:
                models.append(ModelInfo(
                    id=model["name"],
                    created=int(datetime.now().timestamp()),
                ))
        except Exception as e:
            logger.warning(f"Could not list Ollama models: {e}")
    
    return {"data": models}


# --------------------------------------------------------------------------
# Background Tasks
# --------------------------------------------------------------------------

async def execute_fine_tune(job_id: str) -> None:
    """Execute fine-tuning job in background.
    
    Args:
        job_id: Job identifier.
    """
    job = state.jobs.get(job_id)
    if not job:
        return
    
    try:
        job["status"] = "training"
        job["updated_at"] = datetime.now().isoformat()
        
        # Import training components
        from ai_forge.training import TrainingForge, ForgeConfig
        
        config = ForgeConfig(
            model_name=job["config"]["base_model"],
            num_epochs=job["config"]["epochs"],
            learning_rate=job["config"]["learning_rate"],
            pissa_rank=job["config"]["rank"],
            use_pissa=job["config"]["use_pissa"],
            output_dir=f"./output/{job_id}",
        )
        
        forge = TrainingForge(config)
        
        # Load model
        forge.load_model()
        
        # Load training data
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=job["data_path"])["train"]
        
        # Train
        results = forge.train(dataset)
        
        # Save
        forge.save_model(f"./output/{job_id}/final")
        
        job["status"] = "completed"
        job["progress"] = 100.0
        job["loss"] = results.get("train_loss", 0.0)
        
    except Exception as e:
        logger.error(f"Fine-tuning failed for {job_id}: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
    
    job["updated_at"] = datetime.now().isoformat()


# --------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
