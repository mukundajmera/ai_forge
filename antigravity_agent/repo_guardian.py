"""Repo Guardian - Mission Control Agent for AI Forge.

This module provides autonomous orchestration of the fine-tuning
pipeline through an agentic interface compatible with Google Antigravity.

Features:
    - Repository health monitoring
    - Automatic pipeline orchestration
    - Quality gate enforcement
    - Deployment automation

Example:
    >>> guardian = RepoGuardian("/path/to/project")
    >>> report = await guardian.analyze_repository()
    >>> if report.ready_for_training:
    ...     await guardian.start_training_pipeline()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthReport:
    """Repository health report.
    
    Attributes:
        timestamp: Report generation time.
        repository_path: Path to repository.
        total_files: Total files in repository.
        code_files: Number of code files.
        training_data_exists: Whether training data exists.
        training_data_samples: Number of training samples.
        model_exists: Whether a trained model exists.
        last_training: Last training timestamp.
        issues: List of identified issues.
        recommendations: List of recommendations.
        ready_for_training: Whether ready for training.
    """
    
    timestamp: datetime
    repository_path: str
    total_files: int = 0
    code_files: int = 0
    training_data_exists: bool = False
    training_data_samples: int = 0
    model_exists: bool = False
    last_training: Optional[datetime] = None
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    ready_for_training: bool = False
    
    def to_markdown(self) -> str:
        """Convert to markdown report."""
        status_emoji = "✅" if self.ready_for_training else "⚠️"
        
        report = f"""# Repository Health Report {status_emoji}

**Generated:** {self.timestamp.isoformat()}
**Repository:** {self.repository_path}

## Summary
| Metric | Value |
|--------|-------|
| Total Files | {self.total_files} |
| Code Files | {self.code_files} |
| Training Data | {"✅ Yes" if self.training_data_exists else "❌ No"} |
| Training Samples | {self.training_data_samples} |
| Model Exists | {"✅ Yes" if self.model_exists else "❌ No"} |

## Issues
"""
        if self.issues:
            for issue in self.issues:
                report += f"- ⚠️ {issue}\n"
        else:
            report += "No issues found.\n"
        
        report += "\n## Recommendations\n"
        if self.recommendations:
            for rec in self.recommendations:
                report += f"- 💡 {rec}\n"
        else:
            report += "No recommendations.\n"
        
        return report


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution.
    
    Attributes:
        auto_extract_data: Automatically extract training data.
        auto_validate: Automatically validate data.
        auto_train: Automatically start training.
        auto_export: Automatically export to GGUF.
        auto_deploy: Automatically deploy to Ollama.
        quality_threshold: Minimum quality score.
        min_training_samples: Minimum training samples required.
    """
    
    auto_extract_data: bool = True
    auto_validate: bool = True
    auto_train: bool = False  # Requires confirmation
    auto_export: bool = False
    auto_deploy: bool = False
    quality_threshold: float = 0.7
    min_training_samples: int = 100


class RepoGuardian:
    """Mission Control agent for autonomous pipeline orchestration.
    
    RepoGuardian monitors a repository, extracts training data,
    manages fine-tuning, and handles deployment—all autonomously
    or with minimal human oversight.
    
    Attributes:
        project_path: Path to the project repository.
        config: Pipeline configuration.
        
    Example:
        >>> guardian = RepoGuardian("/path/to/my/project")
        >>> 
        >>> # Analyze repository health
        >>> report = await guardian.analyze_repository()
        >>> print(report.to_markdown())
        >>> 
        >>> # Run full pipeline
        >>> if report.ready_for_training:
        ...     result = await guardian.run_pipeline()
    """
    
    def __init__(
        self,
        project_path: str | Path,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """Initialize RepoGuardian.
        
        Args:
            project_path: Path to the project repository.
            config: Pipeline configuration.
        """
        self.project_path = Path(project_path)
        self.config = config or PipelineConfig()
        
        # State
        self._last_report: Optional[HealthReport] = None
        self._pipeline_running = False
        
        if not self.project_path.exists():
            raise ValueError(f"Project path does not exist: {self.project_path}")
        
        logger.info(f"Initialized RepoGuardian for {self.project_path}")
    
    async def analyze_repository(self) -> HealthReport:
        """Analyze repository health and readiness.
        
        Returns:
            HealthReport with analysis results.
        """
        logger.info("Analyzing repository...")
        
        report = HealthReport(
            timestamp=datetime.now(),
            repository_path=str(self.project_path),
        )
        
        # Count files
        report.total_files = sum(1 for _ in self.project_path.rglob("*") if _.is_file())
        
        # Count code files
        code_extensions = {".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c"}
        report.code_files = sum(
            1 for f in self.project_path.rglob("*")
            if f.is_file() and f.suffix in code_extensions
        )
        
        # Check training data
        data_path = self.project_path / "data" / "training_data.json"
        if data_path.exists():
            report.training_data_exists = True
            try:
                import json
                data = json.loads(data_path.read_text())
                report.training_data_samples = len(data)
            except Exception:
                pass
        
        # Check for model
        model_path = self.project_path / "output" / "model"
        report.model_exists = model_path.exists()
        
        # Generate issues and recommendations
        if report.code_files < 10:
            report.issues.append("Very few code files found")
            report.recommendations.append("Add more source code for better training data")
        
        if not report.training_data_exists:
            report.issues.append("No training data found")
            report.recommendations.append("Run data extraction pipeline first")
        elif report.training_data_samples < self.config.min_training_samples:
            report.issues.append(
                f"Insufficient training samples ({report.training_data_samples} < {self.config.min_training_samples})"
            )
            report.recommendations.append("Extract more data or use data augmentation")
        
        # Determine readiness
        report.ready_for_training = (
            report.training_data_exists
            and report.training_data_samples >= self.config.min_training_samples
            and len([i for i in report.issues if "Insufficient" not in i]) == 0
        )
        
        self._last_report = report
        return report
    
    async def extract_training_data(self) -> dict[str, Any]:
        """Extract training data from repository.
        
        Returns:
            Extraction results.
        """
        logger.info("Extracting training data...")
        
        from ai_forge.data_pipeline import CodeMiner, RAFTGenerator, DataValidator
        
        # Mine code
        miner = CodeMiner(self.project_path)
        chunks = miner.extract_all()
        
        # Generate RAFT data
        generator = RAFTGenerator(chunks)
        dataset = generator.generate_dataset()
        
        # Validate
        validator = DataValidator(dataset)
        result = validator.validate_all()
        
        if result.is_valid:
            # Save
            output_path = self.project_path / "data" / "training_data.json"
            output_path.parent.mkdir(exist_ok=True)
            validator.save_cleaned_data(str(output_path))
        
        return {
            "chunks_extracted": len(chunks),
            "samples_generated": len(dataset),
            "samples_valid": result.valid_samples,
            "is_valid": result.is_valid,
            "output_path": str(output_path) if result.is_valid else None,
        }
    
    async def start_training(
        self,
        model_name: str = "unsloth/Llama-3.2-3B-Instruct",
        epochs: int = 3,
    ) -> dict[str, Any]:
        """Start fine-tuning.
        
        Args:
            model_name: Base model to fine-tune.
            epochs: Number of training epochs.
            
        Returns:
            Training results.
        """
        logger.info(f"Starting training with {model_name}...")
        
        from ai_forge.training import TrainingForge, ForgeConfig
        
        config = ForgeConfig(
            model_name=model_name,
            num_epochs=epochs,
            output_dir=str(self.project_path / "output"),
        )
        
        forge = TrainingForge(config)
        forge.load_model()
        
        # Load training data
        data_path = self.project_path / "data" / "training_data.json"
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=str(data_path))["train"]
        
        # Train
        results = forge.train(dataset)
        
        # Save
        forge.save_model(str(self.project_path / "output" / "model"))
        
        return results
    
    async def evaluate_model(self) -> dict[str, float]:
        """Evaluate the trained model.
        
        Returns:
            Evaluation metrics.
        """
        logger.info("Evaluating model...")
        
        from ai_forge.judge import ModelEvaluator
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_path = self.project_path / "output" / "model"
        
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        evaluator = ModelEvaluator(model, tokenizer)
        
        # Load test data
        data_path = self.project_path / "data" / "training_data.json"
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=str(data_path))["train"]
        
        results = evaluator.evaluate_all(dataset)
        
        return results.to_dict()
    
    async def export_model(
        self,
        quantization: str = "q4_k_m",
    ) -> dict[str, Any]:
        """Export model to GGUF format.
        
        Args:
            quantization: Quantization type.
            
        Returns:
            Export results.
        """
        logger.info(f"Exporting model with {quantization} quantization...")
        
        from ai_forge.judge import GGUFExporter, ExportConfig
        
        model_path = self.project_path / "output" / "model"
        
        config = ExportConfig(
            quantization=quantization,
            output_dir=str(self.project_path / "export"),
            model_name=self.project_path.name,
        )
        
        exporter = GGUFExporter(model_path, config)
        result = exporter.export()
        
        return {
            "success": result.success,
            "output_path": str(result.output_path) if result.output_path else None,
            "size_mb": result.model_size_mb,
            "error": result.error,
        }
    
    async def deploy_to_ollama(self) -> dict[str, Any]:
        """Deploy model to Ollama.
        
        Returns:
            Deployment results.
        """
        logger.info("Deploying to Ollama...")
        
        from ai_forge.conductor import OllamaManager
        
        manager = OllamaManager()
        
        # Find GGUF file
        export_dir = self.project_path / "export"
        gguf_files = list(export_dir.glob("*.gguf"))
        
        if not gguf_files:
            return {"success": False, "error": "No GGUF file found"}
        
        gguf_path = gguf_files[0]
        model_name = f"{self.project_path.name}:custom"
        
        from ai_forge.judge import GGUFExporter
        exporter = GGUFExporter(self.project_path / "output" / "model")
        
        success = exporter.deploy_to_ollama(gguf_path, model_name)
        
        return {
            "success": success,
            "model_name": model_name if success else None,
        }
    
    async def run_pipeline(
        self,
        skip_extraction: bool = False,
        skip_training: bool = False,
        skip_export: bool = False,
        skip_deploy: bool = False,
    ) -> dict[str, Any]:
        """Run the full pipeline.
        
        Args:
            skip_extraction: Skip data extraction.
            skip_training: Skip training.
            skip_export: Skip GGUF export.
            skip_deploy: Skip Ollama deployment.
            
        Returns:
            Pipeline results.
        """
        logger.info("Starting full pipeline...")
        
        results: dict[str, Any] = {
            "started_at": datetime.now().isoformat(),
        }
        
        try:
            # Step 1: Analyze
            report = await self.analyze_repository()
            results["analysis"] = {"ready": report.ready_for_training}
            
            # Step 2: Extract data
            if not skip_extraction and self.config.auto_extract_data:
                results["extraction"] = await self.extract_training_data()
            
            # Step 3: Train
            if not skip_training and self.config.auto_train:
                results["training"] = await self.start_training()
            
            # Step 4: Evaluate
            if "training" in results:
                results["evaluation"] = await self.evaluate_model()
            
            # Step 5: Export
            if not skip_export and self.config.auto_export:
                results["export"] = await self.export_model()
            
            # Step 6: Deploy
            if not skip_deploy and self.config.auto_deploy:
                results["deployment"] = await self.deploy_to_ollama()
            
            results["success"] = True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        results["completed_at"] = datetime.now().isoformat()
        return results
