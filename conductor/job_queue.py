"""Job Queue - Async task management for long-running jobs.

This module provides job queue management for fine-tuning tasks
with status tracking and persistence.

Example:
    >>> queue = JobQueue()
    >>> job = await queue.submit(fine_tune_task, config)
    >>> status = await queue.get_status(job.id)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status enumeration."""
    
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents a queued job.
    
    Attributes:
        id: Unique job identifier.
        name: Human-readable job name.
        status: Current job status.
        progress: Progress percentage (0-100).
        result: Job result if completed.
        error: Error message if failed.
        created_at: Creation timestamp.
        started_at: Start timestamp.
        completed_at: Completion timestamp.
        metadata: Additional job metadata.
    """
    
    id: str
    name: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


@dataclass
class JobQueueConfig:
    """Configuration for JobQueue.
    
    Attributes:
        max_concurrent_jobs: Maximum concurrent jobs.
        max_queue_size: Maximum queue size.
        job_timeout_seconds: Job timeout.
        persist_jobs: Whether to persist jobs.
        persist_path: Path for job persistence.
    """
    
    max_concurrent_jobs: int = 1  # Fine-tuning is resource-intensive
    max_queue_size: int = 100
    job_timeout_seconds: int = 86400  # 24 hours
    persist_jobs: bool = True
    persist_path: str = "./jobs.json"


class JobQueue:
    """Async job queue for managing long-running tasks.
    
    Provides queuing, execution, and status tracking for
    fine-tuning jobs and other long-running tasks.
    
    Attributes:
        config: Queue configuration.
        jobs: Dictionary of all jobs.
        
    Example:
        >>> queue = JobQueue()
        >>> job = await queue.submit(my_task, arg1, arg2, name="My Job")
        >>> await queue.wait_for(job.id)
    """
    
    def __init__(self, config: Optional[JobQueueConfig] = None) -> None:
        """Initialize JobQueue.
        
        Args:
            config: Queue configuration.
        """
        self.config = config or JobQueueConfig()
        self.jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[tuple[str, Callable, tuple, dict]] = asyncio.Queue(
            maxsize=self.config.max_queue_size
        )
        self._workers: list[asyncio.Task] = []
        self._running_count = 0
        self._lock = asyncio.Lock()
        self._started = False
        
        logger.info(f"Initialized JobQueue with {self.config.max_concurrent_jobs} workers")
    
    async def start(self) -> None:
        """Start the job queue workers."""
        if self._started:
            return
        
        self._started = True
        
        # Load persisted jobs
        if self.config.persist_jobs:
            await self._load_jobs()
        
        # Start worker tasks
        for i in range(self.config.max_concurrent_jobs):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)
        
        logger.info("JobQueue started")
    
    async def stop(self) -> None:
        """Stop the job queue workers."""
        if not self._started:
            return
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._started = False
        
        # Persist jobs
        if self.config.persist_jobs:
            await self._save_jobs()
        
        logger.info("JobQueue stopped")
    
    async def submit(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Job:
        """Submit a job to the queue.
        
        Args:
            func: Async function to execute.
            *args: Positional arguments for the function.
            name: Optional job name.
            metadata: Optional job metadata.
            **kwargs: Keyword arguments for the function.
            
        Returns:
            Created Job object.
        """
        # Create job
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            name=name or func.__name__,
            status=JobStatus.QUEUED,
            metadata=metadata or {},
        )
        
        self.jobs[job_id] = job
        
        # Queue the job
        await self._queue.put((job_id, func, args, kwargs))
        
        logger.info(f"Submitted job: {job_id} ({job.name})")
        return job
    
    async def get_status(self, job_id: str) -> Optional[Job]:
        """Get job status.
        
        Args:
            job_id: Job identifier.
            
        Returns:
            Job object or None if not found.
        """
        return self.jobs.get(job_id)
    
    async def cancel(self, job_id: str) -> bool:
        """Cancel a job.
        
        Args:
            job_id: Job identifier.
            
        Returns:
            True if cancellation succeeded.
        """
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False
        
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        
        logger.info(f"Cancelled job: {job_id}")
        return True
    
    async def wait_for(
        self,
        job_id: str,
        timeout: Optional[float] = None,
    ) -> Job:
        """Wait for a job to complete.
        
        Args:
            job_id: Job identifier.
            timeout: Optional timeout in seconds.
            
        Returns:
            Completed Job object.
            
        Raises:
            TimeoutError: If timeout exceeded.
            KeyError: If job not found.
        """
        if job_id not in self.jobs:
            raise KeyError(f"Job not found: {job_id}")
        
        start = datetime.now()
        
        while True:
            job = self.jobs[job_id]
            
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return job
            
            if timeout:
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed > timeout:
                    raise TimeoutError(f"Job {job_id} timed out")
            
            await asyncio.sleep(0.5)
    
    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
    ) -> list[Job]:
        """List all jobs.
        
        Args:
            status: Optional status filter.
            
        Returns:
            List of jobs.
        """
        jobs = list(self.jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    def update_progress(self, job_id: str, progress: float) -> None:
        """Update job progress.
        
        Args:
            job_id: Job identifier.
            progress: Progress percentage (0-100).
        """
        job = self.jobs.get(job_id)
        if job:
            job.progress = min(max(progress, 0), 100)
    
    async def _worker(self, worker_id: int) -> None:
        """Worker task that processes jobs.
        
        Args:
            worker_id: Worker identifier.
        """
        logger.debug(f"Worker {worker_id} started")
        
        while True:
            try:
                # Get job from queue
                job_id, func, args, kwargs = await self._queue.get()
                
                job = self.jobs.get(job_id)
                if not job or job.status == JobStatus.CANCELLED:
                    continue
                
                # Update status
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now()
                
                async with self._lock:
                    self._running_count += 1
                
                logger.info(f"Worker {worker_id} starting job: {job_id}")
                
                try:
                    # Execute job with timeout
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.job_timeout_seconds,
                    )
                    
                    job.result = result
                    job.status = JobStatus.COMPLETED
                    job.progress = 100.0
                    
                    logger.info(f"Job completed: {job_id}")
                    
                except asyncio.TimeoutError:
                    job.status = JobStatus.FAILED
                    job.error = "Job timed out"
                    logger.error(f"Job timed out: {job_id}")
                    
                except Exception as e:
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                    logger.error(f"Job failed: {job_id} - {e}")
                
                finally:
                    job.completed_at = datetime.now()
                    
                    async with self._lock:
                        self._running_count -= 1
                    
                    # Persist
                    if self.config.persist_jobs:
                        await self._save_jobs()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    async def _load_jobs(self) -> None:
        """Load persisted jobs."""
        import json
        from pathlib import Path
        
        path = Path(self.config.persist_path)
        if not path.exists():
            return
        
        try:
            data = json.loads(path.read_text())
            for job_data in data:
                job = Job(
                    id=job_data["id"],
                    name=job_data["name"],
                    status=JobStatus(job_data["status"]),
                    progress=job_data.get("progress", 0),
                    result=job_data.get("result"),
                    error=job_data.get("error"),
                    created_at=datetime.fromisoformat(job_data["created_at"]),
                    metadata=job_data.get("metadata", {}),
                )
                self.jobs[job.id] = job
            
            logger.info(f"Loaded {len(self.jobs)} jobs")
        except Exception as e:
            logger.warning(f"Failed to load jobs: {e}")
    
    async def _save_jobs(self) -> None:
        """Save jobs to disk."""
        import json
        from pathlib import Path
        
        path = Path(self.config.persist_path)
        
        try:
            data = [job.to_dict() for job in self.jobs.values()]
            path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save jobs: {e}")
