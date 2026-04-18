"""
Data models and type definitions for Ollama TUI.

All domain objects are defined here with strict typing and Pydantic validation
to ensure data integrity throughout the application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class GPUFitStatus(str, Enum):
    """Describes whether a model fits in GPU VRAM."""

    FITS = "fits"  # Model fully fits in VRAM
    PARTIAL = "partial"  # Partial offload needed (some layers on CPU)
    OOM = "oom"  # Out of memory — full CPU fallback


class QuantizationType(str, Enum):
    """Known quantization levels for GGUF models."""

    Q2_K = "q2_k"
    Q3_K = "q3_k"
    Q4_0 = "q4_0"
    Q4_K = "q4_k"
    Q4_K_M = "q4_k_m"
    Q4_K_S = "q4_k_s"
    Q5_0 = "q5_0"
    Q5_K = "q5_k"
    Q5_K_M = "q5_k_m"
    Q6_K = "q6_k"
    Q8_0 = "q8_0"
    FP16 = "fp16"
    BF16 = "bf16"
    UNKNOWN = "unknown"


class ModelCapability(str, Enum):
    """High-level capability tags for LLM models."""

    REASONING = "reasoning"
    CODING = "coding"
    MULTILINGUAL = "multilingual"
    VISION = "vision"
    EMBEDDING = "embedding"
    CHAT = "chat"
    INSTRUCTION = "instruction"
    RAG = "rag"


class BenchmarkPreset(str, Enum):
    """Pre-defined benchmark prompt types."""

    SHORT = "short"  # Quick single-sentence prompt
    LONG = "long"  # Multi-paragraph prompt
    REASONING = "reasoning"  # Chain-of-thought style prompt
    CODING = "coding"  # Code generation prompt


class OllamaModel(BaseModel):
    """Represents a single model available in the local Ollama instance.

    Attributes:
        name: Full model name (e.g., 'llama3:8b').
        display_name: Human-readable short name.
        param_size_b: Number of parameters in billions.
        quantization: Quantization type (Q4, Q8, FP16, etc.).
        disk_size_gb: Disk space used in gigabytes.
        is_gguf: Whether the model is in GGUF format.
        context_length: Maximum context window in tokens.
        capabilities: List of capability tags.
        family: Model family (e.g., 'llama', 'mistral').
    """

    name: str = Field(..., description="Full Ollama model identifier")
    display_name: str = Field("", description="Short display name")
    param_size_b: float = Field(0.0, ge=0, description="Parameter count in billions")
    quantization: QuantizationType = Field(QuantizationType.UNKNOWN)
    disk_size_gb: float = Field(0.0, ge=0, description="Disk size in GB")
    is_gguf: bool = Field(True, description="Whether model uses GGUF format")
    context_length: int = Field(
        4096, ge=512, description="Max context window in tokens"
    )
    capabilities: list[ModelCapability] = Field(default_factory=list)
    family: str = Field("unknown", description="Model family name")
    modified_at: str = Field("", description="ISO timestamp of last modification")

    @field_validator("display_name", mode="before")
    @classmethod
    def set_display_name(cls, v: str, info) -> str:
        """Use the 'name' field as fallback if display_name is empty."""
        # Fall back to name if not explicitly set
        if not v:
            return info.data.get("name", "unknown")
        return v

    @field_validator("param_size_b", mode="before")
    @classmethod
    def validate_param_size(cls, v) -> float:
        """Coerce param size and clamp negatives to 0."""
        try:
            val = float(v)
            return max(0.0, val)
        except (TypeError, ValueError):
            return 0.0


class VRAMEstimate(BaseModel):
    """VRAM usage estimate for a specific model and configuration.

    Attributes:
        model_name: Reference to the model name.
        context_length: Context length used for this estimate.
        base_vram_gb: VRAM required for model weights only.
        kv_cache_gb: VRAM for KV cache at the given context.
        total_vram_gb: Total estimated VRAM requirement.
        fit_status: Whether it fits in available GPU VRAM.
        offload_layers: Number of layers that must go to CPU.
        total_layers: Total transformer layers in the model.
        cpu_fallback: Whether full CPU fallback is needed.
    """

    model_name: str
    context_length: int = Field(4096, ge=512)
    base_vram_gb: float = Field(0.0, ge=0)
    kv_cache_gb: float = Field(0.0, ge=0)
    total_vram_gb: float = Field(0.0, ge=0)
    fit_status: GPUFitStatus = GPUFitStatus.OOM
    offload_layers: int = Field(0, ge=0)
    total_layers: int = Field(32, ge=1)
    cpu_fallback: bool = False

    @model_validator(mode="after")
    def compute_total(self) -> "VRAMEstimate":
        """Ensure total_vram_gb is sum of components if not set."""
        if self.total_vram_gb == 0.0:
            self.total_vram_gb = self.base_vram_gb + self.kv_cache_gb
        return self


class GPUInfo(BaseModel):
    """GPU hardware information collected from the system.

    Attributes:
        name: GPU model name.
        vram_total_gb: Total VRAM capacity in GB.
        vram_free_gb: Currently free VRAM in GB.
        compute_capability: CUDA compute capability string (e.g., '8.6').
        driver_version: CUDA/driver version string.
        is_available: Whether a compatible GPU was found.
    """

    name: str = Field("Unknown GPU")
    vram_total_gb: float = Field(0.0, ge=0)
    vram_free_gb: float = Field(0.0, ge=0)
    compute_capability: str = Field("", description="CUDA compute capability")
    driver_version: str = Field("")
    is_available: bool = False

    @field_validator("vram_free_gb", mode="after")
    @classmethod
    def cap_free_vram(cls, v: float, info) -> float:
        """Free VRAM cannot exceed total VRAM."""
        total = info.data.get("vram_total_gb", 0.0)
        return min(v, total)


class SystemInfo(BaseModel):
    """Full system hardware snapshot.

    Attributes:
        gpu: GPU information (may have is_available=False on CPU-only systems).
        ram_total_gb: Total system RAM in GB.
        ram_free_gb: Currently free RAM in GB.
        cpu_threads: Number of logical CPU threads.
        cpu_model: CPU model string.
        os_name: Operating system identifier.
    """

    gpu: GPUInfo = Field(default_factory=GPUInfo)
    ram_total_gb: float = Field(0.0, ge=0)
    ram_free_gb: float = Field(0.0, ge=0)
    cpu_threads: int = Field(1, ge=1)
    cpu_model: str = Field("Unknown CPU")
    os_name: str = Field("")


class BenchmarkResult(BaseModel):
    """Benchmark result for a single model run.

    Attributes:
        model_name: The model that was benchmarked.
        preset: Which prompt preset was used.
        tokens_per_sec: Throughput in tokens per second.
        first_token_latency_ms: Time to first token in milliseconds.
        total_latency_ms: Total generation time in milliseconds.
        total_tokens: Number of tokens generated.
        score_label: Human-readable grade (A+, A, B, C, F).
        error: Error message if benchmark failed.
    """

    model_name: str
    preset: BenchmarkPreset
    tokens_per_sec: float = Field(0.0, ge=0)
    first_token_latency_ms: float = Field(0.0, ge=0)
    total_latency_ms: float = Field(0.0, ge=0)
    total_tokens: int = Field(0, ge=0)
    score_label: str = Field("N/A")
    error: Optional[str] = None

    @model_validator(mode="after")
    def compute_score(self) -> "BenchmarkResult":
        """Assign a letter grade based on tokens/sec if not set."""
        if self.score_label == "N/A" and self.tokens_per_sec > 0:
            tps = self.tokens_per_sec
            if tps >= 50:
                self.score_label = "A+"
            elif tps >= 35:
                self.score_label = "A"
            elif tps >= 20:
                self.score_label = "B"
            elif tps >= 10:
                self.score_label = "C"
            else:
                self.score_label = "F"
        return self


class MultiModelSimulation(BaseModel):
    """Simulates loading multiple models concurrently into VRAM.

    Attributes:
        model_names: List of model names to simulate loading together.
        total_vram_needed_gb: Combined VRAM requirement.
        fits_in_gpu: Whether all models fit simultaneously.
        recommendations: Suggested alternatives or actions.
    """

    model_names: list[str] = Field(default_factory=list)
    total_vram_needed_gb: float = Field(0.0, ge=0)
    fits_in_gpu: bool = False
    recommendations: list[str] = Field(default_factory=list)


class RAGPipelineEstimate(BaseModel):
    """Estimates resource usage for a RAG (Retrieval-Augmented Generation) pipeline.

    Attributes:
        embedding_model: Name of the embedding model used.
        generation_model: Name of the LLM used for generation.
        embedding_vram_gb: VRAM for the embedding model.
        generation_vram_gb: VRAM for the generation model.
        retrieval_latency_ms: Estimated retrieval latency.
        rerank_latency_ms: Estimated reranking latency.
        generation_latency_ms: Estimated generation latency.
        total_pipeline_latency_ms: End-to-end pipeline latency.
    """

    embedding_model: str = ""
    generation_model: str = ""
    embedding_vram_gb: float = Field(0.0, ge=0)
    generation_vram_gb: float = Field(0.0, ge=0)
    retrieval_latency_ms: float = Field(50.0, ge=0)
    rerank_latency_ms: float = Field(30.0, ge=0)
    generation_latency_ms: float = Field(0.0, ge=0)
    total_pipeline_latency_ms: float = Field(0.0, ge=0)

    @model_validator(mode="after")
    def compute_total_latency(self) -> "RAGPipelineEstimate":
        """Sum all latency components into total if not provided."""
        if self.total_pipeline_latency_ms == 0.0:
            self.total_pipeline_latency_ms = (
                self.retrieval_latency_ms
                + self.rerank_latency_ms
                + self.generation_latency_ms
            )
        return self
