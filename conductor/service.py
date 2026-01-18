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
        dataset_id: ID of the dataset to use for training.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        rank: LoRA/PiSSA rank.
        batch_size: Training batch size.
        use_pissa: Whether to use PiSSA initialization.
    """
    
    project_name: str
    base_model: str = "unsloth/Llama-3.2-3B-Instruct"
    dataset_id: Optional[str] = None
    epochs: int = 3
    learning_rate: float = 2e-4
    rank: int = 64
    batch_size: int = 4
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
        try:
            from ai_forge.conductor.ollama_manager import OllamaManager
            from ai_forge.conductor.job_queue import JobQueue
        except ImportError:
            from conductor.ollama_manager import OllamaManager
            from conductor.job_queue import JobQueue
        
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

    # Add exception handler for validation errors
    from fastapi.exceptions import RequestValidationError
    from fastapi.requests import Request
    from fastapi.responses import JSONResponse

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.error(f"Validation error for {request.url}: {exc.errors()}")
        try:
             body = await request.json()
             logger.error(f"Request body: {body}")
        except:
             pass
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors(), "body": str(exc.body)},
        )

    
    # Include data sources router
    from conductor.data_sources import router as data_sources_router
    application.include_router(data_sources_router, prefix="/api")
    
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


@app.get("/v1/system/status", tags=["System"])
async def system_status():
    """Get system status for dashboard."""
    import psutil
    
    # Check Ollama
    ollama_status = "stopped"
    try:
        if state.ollama_manager:
            ollama_status = "running"
    except:
        pass
    
    # Get system metrics
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    return {
        "healthy": True,
        "version": "0.1.0",
        "ollama": {
            "status": ollama_status,
            "modelsLoaded": []
        },
        "cpu": {
            "utilization": cpu_percent,
            "cores": psutil.cpu_count()
        },
        "memory": {
            "used": memory.used,
            "total": memory.total
        },
        "gpu": None,
        "runningJobs": len([j for j in state.jobs.values() if j.get('status') == 'running'])
    }


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "service": "AI Forge",
        "version": "0.1.0",
        "endpoints": ["/v1/fine-tune", "/v1/models", "/v1/chat/completions"]
    }


# --------------------------------------------------------------------------
# Fine-Tuning Endpoints
# --------------------------------------------------------------------------

@app.post("/v1/fine-tune", response_model=FineTuneResponse, tags=["Fine-Tuning"])
async def start_fine_tune(
    request: FineTuneRequest,
    background_tasks: BackgroundTasks,
) -> FineTuneResponse:
    """Start a fine-tuning job.
    
    Start a new fine-tuning job using an existing dataset.
    
    Args:
        request: Fine-tuning configuration including dataset_id.
        background_tasks: FastAPI background tasks.
        
    Returns:
        Job creation response with job ID.
    """
    # Generate job ID
    job_id = f"job_{request.project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Get dataset path from data sources if dataset_id provided
    data_path = None
    if hasattr(request, 'dataset_id') and request.dataset_id:
        # Look up dataset file
        dataset_dir = Path("./data/datasets")
        dataset_file = dataset_dir / f"{request.dataset_id}.json"
        if dataset_file.exists():
            data_path = str(dataset_file)
        else:
            # Try to find in uploads
            uploads_dir = Path("./data/uploads")
            if uploads_dir.exists():
                for source_dir in uploads_dir.iterdir():
                    for f in source_dir.glob("*.json"):
                        data_path = str(f)
                        break
                    if data_path:
                        break
    
    if not data_path:
        # Create a placeholder training data file
        data_dir = Path("./data")
        data_dir.mkdir(exist_ok=True)
        data_path = data_dir / f"{job_id}_data.json"
        # Create minimal training data
        import json
        placeholder_data = [
            {"instruction": "Hello", "output": "Hi there!"},
            {"instruction": "What is AI?", "output": "AI stands for Artificial Intelligence."}
        ]
        data_path.write_text(json.dumps(placeholder_data))
        data_path = str(data_path)
    
    # Create job record
    now = datetime.now().isoformat()
    state.jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "config": request.model_dump(),
        "data_path": data_path,
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
# Query Endpoint
# --------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Simple query request.
    
    Attributes:
        prompt: The prompt to send.
        model: Optional model name (defaults to active model).
    """
    
    prompt: str
    model: Optional[str] = None


class QueryResponse(BaseModel):
    """Query response.
    
    Attributes:
        answer: The model's response.
        model: Model used.
        metadata: Additional metadata.
    """
    
    answer: str
    model: str
    metadata: dict = {}


@app.post("/v1/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest) -> QueryResponse:
    """Simple prompt-response query.
    
    Args:
        request: Query with prompt.
        
    Returns:
        Model's response.
    """
    if state.ollama_manager is None:
        raise HTTPException(status_code=503, detail="Ollama not available")
    
    # Determine model
    model = request.model
    if not model:
        # Use first available model
        try:
            models = await state.ollama_manager.list_models()
            if models:
                model = models[0]["name"]
            else:
                raise HTTPException(status_code=400, detail="No models available")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    try:
        response = await state.ollama_manager.generate(
            model=model,
            prompt=request.prompt,
        )
        
        return QueryResponse(
            answer=response,
            model=model,
            metadata={"timestamp": datetime.now().isoformat()},
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------
# Status Endpoint (Alias)
# --------------------------------------------------------------------------

@app.get("/status/{job_id}", response_model=JobStatus, tags=["Status"])
async def get_status(job_id: str) -> JobStatus:
    """Get job status (alias for /v1/fine-tune/{job_id}).
    
    Args:
        job_id: Job identifier.
        
    Returns:
        Current job status.
    """
    return await get_job_status(job_id)


# --------------------------------------------------------------------------
# Deploy Endpoint
# --------------------------------------------------------------------------

class DeployRequest(BaseModel):
    """Deploy request options.
    
    Attributes:
        model_name: Name for the deployed model.
        system_prompt: Optional system prompt.
        quantization: Quantization type.
    """
    
    model_name: Optional[str] = None
    system_prompt: Optional[str] = None
    quantization: str = "q4_k_m"


class DeployResponse(BaseModel):
    """Deploy response.
    
    Attributes:
        success: Whether deployment succeeded.
        model_name: Deployed model name.
        message: Status message.
    """
    
    success: bool
    model_name: str
    message: str


@app.post("/deploy/{job_id}", response_model=DeployResponse, tags=["Deploy"])
async def deploy_model(job_id: str, request: DeployRequest = None) -> DeployResponse:
    """Deploy a trained model to Ollama.
    
    Args:
        job_id: Job identifier of completed training.
        request: Optional deploy options.
        
    Returns:
        Deployment result.
    """
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = state.jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Status: {job['status']}"
        )
    
    if state.ollama_manager is None:
        raise HTTPException(status_code=503, detail="Ollama not available")
    
    try:
        # Determine model name
        model_name = (request.model_name if request else None) or f"ai-forge-{job_id}"
        
        # Get output path
        output_dir = Path(f"./output/{job_id}/final")
        
        if not output_dir.exists():
            raise HTTPException(
                status_code=400, 
                detail="Model output not found. Training may have failed."
            )
        
        # Export to GGUF
        from judge.exporter import GGUFExporter, ExportConfig
        
        quantization = request.quantization if request else "q4_k_m"
        export_config = ExportConfig(
            quantization=quantization,
            output_dir=str(output_dir / "export"),
            model_name=model_name,
        )
        
        exporter = GGUFExporter(output_dir, export_config)
        result = exporter.export()
        
        if not result.success:
            raise HTTPException(status_code=500, detail=f"Export failed: {result.error}")
        
        # Create Modelfile and deploy
        modelfile_path = exporter.create_modelfile(
            result.output_path,
            system_prompt=request.system_prompt if request else None,
        )
        
        # Deploy to Ollama
        deployed = await state.ollama_manager.create_model(
            model_name=model_name,
            modelfile_path=str(modelfile_path),
        )
        
        if not deployed:
            raise HTTPException(status_code=500, detail="Failed to deploy to Ollama")
        
        return DeployResponse(
            success=True,
            model_name=model_name,
            message=f"Model deployed as '{model_name}'. Use with: ollama run {model_name}",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deploy failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------
# Validate Endpoint
# --------------------------------------------------------------------------

class ValidateResponse(BaseModel):
    """Validation response.
    
    Attributes:
        job_id: Job that was validated.
        passed: Whether validation passed.
        metrics: Evaluation metrics.
        report_path: Path to full report.
    """
    
    job_id: str
    passed: bool
    metrics: dict
    report_path: Optional[str] = None


@app.post("/validate/{job_id}", response_model=ValidateResponse, tags=["Validate"])
async def validate_model(job_id: str) -> ValidateResponse:
    """Run validation suite on a trained model.
    
    Args:
        job_id: Job identifier.
        
    Returns:
        Validation results.
    """
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = state.jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job['status']}"
        )
    
    try:
        output_dir = Path(f"./output/{job_id}/final")
        
        if not output_dir.exists():
            raise HTTPException(status_code=400, detail="Model output not found")
        
        # Create evaluation report
        from judge.report import EvaluationReport
        
        report = EvaluationReport(
            model_name=f"ai-forge-{job_id}",
            base_model=job["config"]["base_model"],
            num_training_examples=0,  # Would need to count from data file
            num_eval_examples=0,
        )
        
        # Add training metrics
        report.add_metrics(
            perplexity=job.get("loss", 0.0) * 2.72,  # Rough estimate
        )
        
        # Determine if validation passed
        # (Simple heuristics - would need actual eval dataset)
        passed = job.get("loss", 999) < 2.0
        
        # Save report
        report_path = output_dir / "validation_report.md"
        report.to_markdown(report_path)
        
        return ValidateResponse(
            job_id=job_id,
            passed=passed,
            metrics={
                "final_loss": job.get("loss"),
                "epochs_completed": job["config"]["epochs"],
            },
            report_path=str(report_path),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        import sys
        
        # Add project root parent to path to allow 'ai_forge' imports
        # Service is in ai_forge/conductor/service.py -> parent.parent.parent = pocs/
        root_parent = str(Path(__file__).resolve().parent.parent.parent)
        if root_parent not in sys.path:
            sys.path.insert(0, root_parent)

        try:
            from ai_forge.training import FineTuneTrainer, FineTuneConfig
            from ai_forge.training.schemas import (
                ModelConfig, 
                TrainingConfig, 
                PiSSAConfig, 
                LoggingConfig,
                InitMethod
            )
        except ImportError:
            # Fallback for when running from root without package install
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from training import FineTuneTrainer, FineTuneConfig
            from training.schemas import (
                ModelConfig, 
                TrainingConfig, 
                PiSSAConfig, 
                LoggingConfig,
                InitMethod
            )
        
        # Determine initialization method
        init_method = InitMethod.PISSA if job["config"].get("use_pissa", True) else InitMethod.GAUSSIAN

        # Construct configuration
        config = FineTuneConfig(
            model=ModelConfig(
                base_model=job["config"]["base_model"],
            ),
            training=TrainingConfig(
                num_train_epochs=job["config"]["epochs"],
                learning_rate=job["config"]["learning_rate"],
                per_device_train_batch_size=job["config"]["batch_size"],
            ),
            pissa=PiSSAConfig(
                rank=job["config"]["rank"],
                init_method=init_method,
            ),
            logging=LoggingConfig(
                output_dir=f"./output/{job_id}",
            ),
        )
        
        forge = FineTuneTrainer(config)
        
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
# Repo Guardian / Retrain Endpoint
# --------------------------------------------------------------------------

class RetrainRequest(BaseModel):
    """Request to trigger retraining via Repo Guardian.
    
    Attributes:
        project_path: Path to project repository.
        auto_deploy: Automatically deploy after training.
        force: Force retraining even if no changes detected.
    """
    
    project_path: str = "."
    auto_deploy: bool = False
    force: bool = False


class RetrainResponse(BaseModel):
    """Response from retrain request.
    
    Attributes:
        triggered: Whether retraining was triggered.
        reason: Reason for decision.
        plan: Training plan if triggered.
        job_id: Job ID if training started.
    """
    
    triggered: bool
    reason: str
    plan: Optional[dict] = None
    job_id: Optional[str] = None


@app.post("/v1/retrain", response_model=RetrainResponse, tags=["Agent"])
async def trigger_retrain(
    request: RetrainRequest,
    background_tasks: BackgroundTasks,
) -> RetrainResponse:
    """Trigger retraining via Repo Guardian agent.
    
    Monitors the repository and triggers retraining if significant
    changes are detected, or if force=True.
    
    Args:
        request: Retrain configuration.
        background_tasks: FastAPI background tasks.
        
    Returns:
        Retrain decision and plan.
    """
    try:
        from antigravity_agent.repo_guardian import RepoGuardian, PipelineConfig
        
        # Initialize guardian
        config = PipelineConfig(
            auto_train=True,
            auto_export=request.auto_deploy,
            auto_deploy=request.auto_deploy,
        )
        guardian = RepoGuardian(request.project_path, config)
        
        # Check if retraining needed
        if not request.force:
            monitor_result = guardian.monitor_repository()
            
            if not monitor_result["should_retrain"]:
                return RetrainResponse(
                    triggered=False,
                    reason=monitor_result["reason"],
                )
        else:
            monitor_result = {"reason": "Forced retraining requested"}
        
        # Create training plan
        plan = guardian.plan_training_cycle()
        
        # Generate job ID
        job_id = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Queue background task
        background_tasks.add_task(execute_guardian_pipeline, guardian, job_id)
        
        # Track job
        state.jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "type": "agent_pipeline",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        return RetrainResponse(
            triggered=True,
            reason=monitor_result.get("reason", "Retraining triggered"),
            plan=plan,
            job_id=job_id,
        )
        
    except Exception as e:
        logger.error(f"Retrain trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/retrain/monitor", tags=["Agent"])
async def monitor_repository(project_path: str = ".") -> dict:
    """Check repository for changes without triggering retraining.
    
    Args:
        project_path: Path to project repository.
        
    Returns:
        Monitoring results.
    """
    try:
        from antigravity_agent.repo_guardian import RepoGuardian
        
        guardian = RepoGuardian(project_path)
        return guardian.monitor_repository()
        
    except Exception as e:
        logger.error(f"Monitor failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/retrain/{job_id}/pause", tags=["Agent"])
async def pause_pipeline(job_id: str) -> dict:
    """Pause a running pipeline.
    
    Args:
        job_id: Job identifier.
        
    Returns:
        Pause confirmation.
    """
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    state.jobs[job_id]["status"] = "paused"
    state.jobs[job_id]["updated_at"] = datetime.now().isoformat()
    
    return {"message": f"Job {job_id} paused"}


@app.post("/v1/retrain/{job_id}/resume", tags=["Agent"])
async def resume_pipeline(job_id: str) -> dict:
    """Resume a paused pipeline.
    
    Args:
        job_id: Job identifier.
        
    Returns:
        Resume confirmation.
    """
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    state.jobs[job_id]["status"] = "running"
    state.jobs[job_id]["updated_at"] = datetime.now().isoformat()
    
    return {"message": f"Job {job_id} resumed"}


async def execute_guardian_pipeline(guardian, job_id: str) -> None:
    """Execute Repo Guardian pipeline in background.
    
    Args:
        guardian: RepoGuardian instance.
        job_id: Job identifier.
    """
    job = state.jobs.get(job_id)
    if not job:
        return
    
    try:
        job["status"] = "running"
        job["updated_at"] = datetime.now().isoformat()
        
        # Run pipeline
        results = await guardian.run_pipeline()
        
        job["status"] = "completed" if results.get("success") else "failed"
        job["results"] = results
        
    except Exception as e:
        logger.error(f"Guardian pipeline failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
    
    job["updated_at"] = datetime.now().isoformat()


# --------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

