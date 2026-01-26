"""GGUF Exporter - Model conversion for Ollama deployment.

This module provides GGUF format conversion for deploying
fine-tuned models to Ollama.

Example:
    >>> exporter = GGUFExporter(model_path="./output/model")
    >>> exporter.export("./export/model.gguf", quantization="q4_k_m")
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)


# Quantization types supported by llama.cpp
QuantizationType = Literal[
    "q4_0", "q4_1", "q5_0", "q5_1",
    "q8_0", "q2_k", "q3_k_s", "q3_k_m", "q3_k_l",
    "q4_k_s", "q4_k_m", "q5_k_s", "q5_k_m",
    "q6_k", "f16", "f32"
]


@dataclass
class ExportConfig:
    """Configuration for GGUF export.
    
    Attributes:
        quantization: Quantization type for GGUF.
        output_dir: Directory to save exported model.
        model_name: Name for the exported model.
        description: Model description for metadata.
        author: Author name for metadata.
        include_vocab: Whether to include vocabulary.
        use_llama_cpp: Whether to use llama.cpp for conversion.
    """
    
    quantization: QuantizationType = "q8_0"
    output_dir: str = "./export"
    model_name: str = "custom_model"
    description: str = "Fine-tuned model with AI Forge"
    author: str = "AI Forge"
    include_vocab: bool = True
    use_llama_cpp: bool = True


@dataclass
class ExportResult:
    """Result from GGUF export.
    
    Attributes:
        success: Whether export succeeded.
        output_path: Path to exported model.
        model_size_mb: Size of exported model in MB.
        quantization: Quantization used.
        error: Error message if failed.
    """
    
    success: bool
    output_path: Optional[Path] = None
    model_size_mb: Optional[float] = None
    quantization: Optional[str] = None
    error: Optional[str] = None


def merge_adapters_to_base(
    base_model_path: str | Path,
    adapter_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Merge PiSSA/LoRA adapters back into base model.
    
    Produces a standalone model that doesn't require adapters at inference.
    
    Args:
        base_model_path: Path to base model or HF model ID.
        adapter_path: Path to trained adapter weights.
        output_path: Optional output path (defaults to adapter_path/merged).
        
    Returns:
        Path to merged model.
        
    Example:
        >>> merged_path = merge_adapters_to_base(
        ...     "unsloth/Llama-3.2-3B-Instruct",
        ...     "./output/adapters"
        ... )
    """
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    adapter_path = Path(adapter_path)
    
    if output_path is None:
        output_path = adapter_path / "merged"
    output_path = Path(output_path)
    
    logger.info(f"Merging adapters from {adapter_path} into {base_model_path}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        # Load base model
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )
        
        # Load adapter
        logger.info("Loading adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Merge adapter into base model
        logger.info("Merging adapter weights...")
        model = model.merge_and_unload()
        
        # Save merged model
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        logger.info(f"Merged model saved to {output_path}")
        return output_path
        
    except ImportError as e:
        logger.error(f"Missing dependencies for adapter merging: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to merge adapters: {e}")
        raise


class GGUFExporter:
    """GGUF format exporter for Ollama deployment.
    
    Converts HuggingFace models to GGUF format compatible
    with llama.cpp and Ollama.
    
    Attributes:
        model_path: Path to the model to export.
        config: Export configuration.
        
    Example:
        >>> exporter = GGUFExporter("./output/my_model")
        >>> result = exporter.export(quantization="q4_k_m")
        >>> if result.success:
        ...     print(f"Exported to {result.output_path}")
    """
    
    # Recommended quantizations for different use cases
    QUANT_RECOMMENDATIONS = {
        "quality": "q5_k_m",      # Best quality
        "balanced": "q4_k_m",     # Quality/size balance
        "speed": "q4_0",          # Fastest inference
        "tiny": "q2_k",           # Smallest size
    }
    
    def __init__(
        self,
        model_path: str | Path,
        config: Optional[ExportConfig] = None,
    ) -> None:
        """Initialize GGUFExporter.
        
        Args:
            model_path: Path to HuggingFace model.
            config: Export configuration.
        """
        self.model_path = Path(model_path)
        self.config = config or ExportConfig()
        
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized GGUFExporter for {self.model_path}")
    
    def _check_dependencies(self) -> bool:
        """Check if required tools are installed.
        
        Returns:
            True if all dependencies available.
        """
        # Check for llama.cpp's convert script
        try:
            result = subprocess.run(
                ["which", "llama-cpp-convert"],
                capture_output=True,
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass
        
        # Check for Python package
        try:
            import llama_cpp
            return True
        except ImportError:
            pass
        
        logger.warning("llama.cpp tools not found")
        return False
    
    def export(
        self,
        output_path: Optional[str] = None,
        quantization: Optional[QuantizationType] = None,
    ) -> ExportResult:
        """Export model to GGUF format.
        
        Args:
            output_path: Optional custom output path.
            quantization: Quantization type to use.
            
        Returns:
            ExportResult with export status.
        """
        quant = quantization or self.config.quantization
        
        if output_path:
            output = Path(output_path)
        else:
            output = Path(self.config.output_dir) / f"{self.config.model_name}-{quant}.gguf"
        
        logger.info(f"Exporting to {output} with {quant} quantization")
        
        try:
            # Step 1: Convert to GGUF
            success = self._convert_to_gguf(output, quant)
            
            if not success:
                return ExportResult(
                    success=False,
                    error="GGUF conversion failed",
                )
            
            # Get file size
            size_mb = output.stat().st_size / (1024 * 1024)
            
            logger.info(f"Export complete: {output} ({size_mb:.1f} MB)")
            
            return ExportResult(
                success=True,
                output_path=output,
                model_size_mb=size_mb,
                quantization=quant,
            )
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ExportResult(
                success=False,
                error=str(e),
            )
    
    def _convert_to_gguf(self, output_path: Path, quantization: str) -> bool:
        """Convert HuggingFace model to GGUF.
        
        Args:
            output_path: Path for output GGUF file.
            quantization: Quantization type.
            
        Returns:
            True if conversion succeeded.
        """
        # Try llama.cpp convert script
        convert_script = self._find_convert_script()
        
        if convert_script:
            return self._convert_with_script(convert_script, output_path, quantization)
        
        # Fallback to Python conversion
        return self._convert_with_python(output_path, quantization)
    
    def _find_convert_script(self) -> Optional[Path]:
        """Find llama.cpp convert script.
        
        Returns:
            Path to convert script or None.
        """
        # Common locations
        search_paths = [
            Path.home() / "llama.cpp" / "convert.py",
            Path.home() / "llama.cpp" / "convert-hf-to-gguf.py",
            Path("./scripts/convert_hf_to_gguf.py").absolute(),
            Path("/usr/local/bin/llama-cpp-convert"),
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        return None
    
    def _convert_with_script(
        self,
        script_path: Path,
        output_path: Path,
        quantization: str,
    ) -> bool:
        """Convert using llama.cpp script.
        
        Args:
            script_path: Path to convert script.
            output_path: Output path.
            quantization: Quantization type.
            
        Returns:
            True if conversion succeeded.
        """
        import sys
        
        cmd = [
            sys.executable,
            str(script_path),
            str(self.model_path),
            "--outfile", str(output_path),
            "--outtype", quantization,
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            logger.error(f"Conversion failed: {result.stderr}")
            return False
        
        return True
    
    def _convert_with_python(self, output_path: Path, quantization: str) -> bool:
        """Convert using Python (fallback method).
        
        Args:
            output_path: Output path.
            quantization: Quantization type.
            
        Returns:
            True if conversion succeeded.
        """
        # This is a placeholder - actual implementation would require
        # significant code to handle GGUF format
        logger.warning(
            "Python GGUF conversion not fully implemented. "
            "Please install llama.cpp for proper conversion."
        )
        return False
    
    def create_modelfile(
        self,
        gguf_path: str | Path,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_k: int = 40,
    ) -> Path:
        """Create Ollama Modelfile.
        
        Args:
            gguf_path: Path to GGUF model.
            system_prompt: System prompt for the model.
            temperature: Default temperature.
            top_k: Default top_k.
            
        Returns:
            Path to created Modelfile.
        """
        gguf_path = Path(gguf_path)
        modelfile_path = gguf_path.parent / "Modelfile"
        
        system_prompt = system_prompt or (
            f"You are a helpful assistant fine-tuned with AI Forge. "
            f"Provide accurate, helpful responses based on your training."
        )
        
        content = f'''FROM {gguf_path.resolve()}

SYSTEM """
{system_prompt}
"""

PARAMETER temperature {temperature}
PARAMETER top_k {top_k}
'''
        
        modelfile_path.write_text(content)
        logger.info(f"Created Modelfile at {modelfile_path}")
        
        return modelfile_path
    
    def deploy_to_ollama(
        self,
        gguf_path: str | Path,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> bool:
        """Deploy model to Ollama.
        
        Args:
            gguf_path: Path to GGUF model.
            model_name: Name for Ollama model.
            system_prompt: System prompt.
            
        Returns:
            True if deployment succeeded.
        """
        gguf_path = Path(gguf_path)
        model_name = model_name or self.config.model_name
        
        # Create Modelfile
        modelfile = self.create_modelfile(gguf_path, system_prompt)
        
        # Create Ollama model
        cmd = ["ollama", "create", model_name, "-f", str(modelfile)]
        
        logger.info(f"Creating Ollama model: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            logger.error(f"Ollama create failed: {result.stderr}")
            return False
        
        logger.info(f"Successfully deployed {model_name} to Ollama")
        return True
    
    def estimate_size(self, quantization: QuantizationType) -> float:
        """Estimate exported model size.
        
        Args:
            quantization: Quantization type.
            
        Returns:
            Estimated size in MB.
        """
        # Load model config to get parameter count
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_path)
            
            # Rough estimate based on architecture
            if hasattr(config, "num_parameters"):
                params = config.num_parameters
            else:
                # Estimate from config
                hidden_size = getattr(config, "hidden_size", 4096)
                num_layers = getattr(config, "num_hidden_layers", 32)
                vocab_size = getattr(config, "vocab_size", 32000)
                
                params = num_layers * hidden_size * hidden_size * 4 + vocab_size * hidden_size
        except Exception:
            params = 3_000_000_000  # Default 3B estimate
        
        # Bits per parameter for different quantization
        bits_per_param = {
            "f32": 32, "f16": 16,
            "q8_0": 8,
            "q6_k": 6.5,
            "q5_k_m": 5.5, "q5_k_s": 5.5, "q5_1": 5.5, "q5_0": 5.5,
            "q4_k_m": 4.5, "q4_k_s": 4.5, "q4_1": 4.5, "q4_0": 4.0,
            "q3_k_l": 3.5, "q3_k_m": 3.4, "q3_k_s": 3.3,
            "q2_k": 2.5,
        }
        
        bits = bits_per_param.get(quantization, 4.5)
        size_bytes = params * bits / 8
        size_mb = size_bytes / (1024 * 1024)
        
        return size_mb
