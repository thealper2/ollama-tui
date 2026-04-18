"""
Model recommendation and advanced analysis engine for Ollama TUI.

Provides auto-recommendations based on system hardware, use-case scenario
matching, quantization impact analysis, and RAG pipeline estimation.
"""

from __future__ import annotations

from typing import Optional

from models import (
    BenchmarkPreset,
    BenchmarkResult,
    ModelCapability,
    OllamaModel,
    QuantizationType,
    RAGPipelineEstimate,
    SystemInfo,
    VRAMEstimate,
)
from utils.logger import setup_logger
from vram_calculator import compute_base_vram, estimate_vram

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Scenario → required capability mapping
# ---------------------------------------------------------------------------
SCENARIO_CAPABILITIES: dict[str, list[ModelCapability]] = {
    "chatbot": [ModelCapability.CHAT, ModelCapability.INSTRUCTION],
    "rag": [ModelCapability.RAG, ModelCapability.INSTRUCTION, ModelCapability.CHAT],
    "coding": [ModelCapability.CODING, ModelCapability.INSTRUCTION],
    "reasoning": [ModelCapability.REASONING],
    "multilingual": [ModelCapability.MULTILINGUAL],
    "vision": [ModelCapability.VISION],
    "embedding": [ModelCapability.EMBEDDING],
}

# Quantization quality proxies (higher = better quality, lower = faster/smaller)
QUANT_QUALITY_SCORE: dict[QuantizationType, int] = {
    QuantizationType.FP16: 100,
    QuantizationType.BF16: 100,
    QuantizationType.Q8_0: 90,
    QuantizationType.Q6_K: 85,
    QuantizationType.Q5_K_M: 78,
    QuantizationType.Q5_K: 75,
    QuantizationType.Q5_0: 73,
    QuantizationType.Q4_K_M: 70,
    QuantizationType.Q4_K: 67,
    QuantizationType.Q4_K_S: 65,
    QuantizationType.Q4_0: 63,
    QuantizationType.Q3_K: 50,
    QuantizationType.Q2_K: 35,
    QuantizationType.UNKNOWN: 60,
}

QUANT_SPEED_SCORE: dict[QuantizationType, int] = {
    QuantizationType.FP16: 50,
    QuantizationType.BF16: 50,
    QuantizationType.Q8_0: 65,
    QuantizationType.Q6_K: 72,
    QuantizationType.Q5_K_M: 78,
    QuantizationType.Q5_K: 80,
    QuantizationType.Q5_0: 82,
    QuantizationType.Q4_K_M: 88,
    QuantizationType.Q4_K: 90,
    QuantizationType.Q4_K_S: 91,
    QuantizationType.Q4_0: 92,
    QuantizationType.Q3_K: 95,
    QuantizationType.Q2_K: 99,
    QuantizationType.UNKNOWN: 80,
}

# Average embedding latency by model size (ms per query)
EMBEDDING_LATENCY_MS: dict[float, float] = {
    0.1: 2.0,
    0.3: 5.0,
    0.5: 8.0,
    1.0: 15.0,
}


def _score_model_for_system(
    model: OllamaModel,
    system_info: SystemInfo,
    context_length: int = 4096,
) -> float:
    """Compute a composite score for how well a model suits the system.

    Scoring factors:
    - VRAM fit: 50 points max
    - Speed (quantization proxy): 30 points max
    - Quality (quantization proxy): 20 points max

    Args:
        model: OllamaModel to score.
        system_info: Hardware information.
        context_length: Context window to assume for VRAM check.

    Returns:
        Float score 0–100.
    """
    vram = system_info.gpu.vram_total_gb if system_info.gpu.is_available else 0.0
    est = estimate_vram(model, context_length, vram)

    # VRAM fit scoring
    from models import GPUFitStatus

    if est.fit_status == GPUFitStatus.FITS:
        vram_score = 50
    elif est.fit_status == GPUFitStatus.PARTIAL:
        vram_score = 25
    else:
        vram_score = 0

    speed_score = QUANT_SPEED_SCORE.get(model.quantization, 80) * 0.30
    quality_score = QUANT_QUALITY_SCORE.get(model.quantization, 60) * 0.20

    return vram_score + speed_score + quality_score


def recommend_models(
    models: list[OllamaModel],
    system_info: SystemInfo,
    scenario: str = "chatbot",
    top_n: int = 3,
) -> list[tuple[OllamaModel, float, str]]:
    """Recommend the best models for a given use-case scenario.

    Args:
        models: All available models to consider.
        system_info: Hardware information.
        scenario: Use-case scenario key ('chatbot', 'rag', 'coding', etc.).
        top_n: Number of top recommendations to return.

    Returns:
        List of (model, score, reason_string) tuples sorted by score descending.
    """
    required_caps = SCENARIO_CAPABILITIES.get(scenario.lower(), [])
    scored: list[tuple[OllamaModel, float, str]] = []

    for model in models:
        # Filter by capability: prefer models with at least one matching capability
        cap_match = any(cap in model.capabilities for cap in required_caps)
        base_score = _score_model_for_system(model, system_info)

        # Boost score for capability match
        if cap_match:
            base_score += 15

        # Build human-readable reason string
        vram = system_info.gpu.vram_total_gb if system_info.gpu.is_available else 0.0
        est = estimate_vram(model, 4096, vram)
        from models import GPUFitStatus

        fit_str = {
            GPUFitStatus.FITS: "✅ fits in VRAM",
            GPUFitStatus.PARTIAL: "⚠ partial GPU offload",
            GPUFitStatus.OOM: "❌ CPU fallback",
        }[est.fit_status]

        quality = QUANT_QUALITY_SCORE.get(model.quantization, 60)
        reason = (
            f"{fit_str} | "
            f"~{est.total_vram_gb:.1f} GB VRAM | "
            f"quality score: {quality}/100"
        )

        scored.append((model, round(base_score, 1), reason))

    # Sort by score descending, then by param size (prefer larger = more capable)
    scored.sort(key=lambda x: (x[1], x[0].param_size_b), reverse=True)
    return scored[:top_n]


def analyze_quantization_impact(
    models: list[OllamaModel],
    system_info: SystemInfo,
) -> list[dict]:
    """Group models by base family and compare quantization variants.

    Finds models that share the same base (e.g., 'llama3:8b-q4' vs 'llama3:8b-q8')
    and produces a comparison showing VRAM vs quality tradeoffs.

    Args:
        models: Full list of available models.
        system_info: System hardware info.

    Returns:
        List of dicts, each representing a family group with variant comparisons.
    """
    # Group by base name (strip quantization suffix)
    import re

    groups: dict[str, list[OllamaModel]] = {}
    for model in models:
        # Strip quantization from name to get base key
        base = re.sub(r"[-_](q\d|fp16|bf16|f16)[^:]*$", "", model.name.lower())
        base = (
            base.split(":")[0] + ":" + model.name.split(":")[1].split("-")[0]
            if ":" in model.name
            else base
        )
        groups.setdefault(base, []).append(model)

    results = []
    vram = system_info.gpu.vram_total_gb if system_info.gpu.is_available else 0.0

    for base_name, variants in groups.items():
        if len(variants) < 2:
            continue  # Only show groups with multiple quantizations

        variant_data = []
        for model in sorted(
            variants, key=lambda m: QUANT_QUALITY_SCORE.get(m.quantization, 60)
        ):
            est = estimate_vram(model, 4096, vram)
            variant_data.append(
                {
                    "name": model.name,
                    "quantization": model.quantization.value,
                    "vram_gb": est.total_vram_gb,
                    "quality_score": QUANT_QUALITY_SCORE.get(model.quantization, 60),
                    "speed_score": QUANT_SPEED_SCORE.get(model.quantization, 80),
                    "fit_status": est.fit_status.value,
                }
            )

        results.append(
            {
                "base": base_name,
                "variants": variant_data,
            }
        )

    return results


def estimate_rag_pipeline(
    embedding_model: Optional[OllamaModel],
    generation_model: Optional[OllamaModel],
    system_info: SystemInfo,
    benchmark_result: Optional[BenchmarkResult] = None,
) -> RAGPipelineEstimate:
    """Estimate total resource usage for a RAG pipeline.

    Combines embedding model VRAM, generation model VRAM,
    and latency estimates into a single pipeline summary.

    Args:
        embedding_model: The embedding model (can be None for no embedding).
        generation_model: The LLM used for generation (can be None).
        system_info: Hardware info for VRAM calculations.
        benchmark_result: Optional real benchmark to use for generation latency.

    Returns:
        RAGPipelineEstimate with combined resource and latency estimates.
    """
    vram_available = (
        system_info.gpu.vram_total_gb if system_info.gpu.is_available else 0.0
    )

    emb_vram = 0.0
    emb_name = ""
    if embedding_model:
        emb_vram = compute_base_vram(embedding_model)
        emb_name = embedding_model.name

    gen_vram = 0.0
    gen_name = ""
    if generation_model:
        est = estimate_vram(generation_model, 4096, vram_available)
        gen_vram = est.total_vram_gb
        gen_name = generation_model.name

    # Retrieval latency: assume vector store lookup (approximate)
    retrieval_ms = 50.0  # Typical in-memory FAISS latency

    # Reranking latency: small model pass
    rerank_ms = 30.0

    # Generation latency: use benchmark or estimate from tokens/sec
    if benchmark_result and benchmark_result.total_latency_ms > 0:
        gen_latency_ms = benchmark_result.total_latency_ms
    elif generation_model and generation_model.param_size_b > 0:
        # Rough estimate: larger models = slower
        gen_latency_ms = generation_model.param_size_b * 15.0  # ms per B params
    else:
        gen_latency_ms = 500.0

    # Embedding lookup latency
    if embedding_model:
        emb_param = max(embedding_model.param_size_b, 0.1)
        emb_key = min(EMBEDDING_LATENCY_MS.keys(), key=lambda k: abs(k - emb_param))
        retrieval_ms += EMBEDDING_LATENCY_MS[emb_key]

    return RAGPipelineEstimate(
        embedding_model=emb_name,
        generation_model=gen_name,
        embedding_vram_gb=round(emb_vram, 2),
        generation_vram_gb=round(gen_vram, 2),
        retrieval_latency_ms=round(retrieval_ms, 1),
        rerank_latency_ms=round(rerank_ms, 1),
        generation_latency_ms=round(gen_latency_ms, 1),
    )


def estimate_prompt_cost(
    tokens: int,
    tokens_per_sec: float,
) -> dict[str, float]:
    """Estimate time-based "cost" for a given number of tokens.

    Since we're running locally, cost is measured in wall-clock time
    rather than money.

    Args:
        tokens: Number of tokens to generate.
        tokens_per_sec: Measured throughput.

    Returns:
        Dict with 'seconds', 'minutes', and 'tokens_per_sec' keys.
    """
    if tokens_per_sec <= 0:
        return {"seconds": 0.0, "minutes": 0.0, "tokens_per_sec": 0.0}
    seconds = tokens / tokens_per_sec
    return {
        "seconds": round(seconds, 2),
        "minutes": round(seconds / 60, 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
    }
