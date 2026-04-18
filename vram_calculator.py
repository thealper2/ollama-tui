"""
VRAM estimation engine for Ollama TUI.

Calculates estimated VRAM usage per model including model weights,
KV cache at various context lengths, and determines GPU fit status.
Also supports multi-model load simulation.
"""

from __future__ import annotations

from typing import Optional

from models import (
    GPUFitStatus,
    MultiModelSimulation,
    OllamaModel,
    QuantizationType,
    SystemInfo,
    VRAMEstimate,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Bits-per-weight lookup for each quantization format
# These values are approximate averages for mixed-quantization GGUF files.
# ---------------------------------------------------------------------------
BITS_PER_WEIGHT: dict[QuantizationType, float] = {
    QuantizationType.Q2_K: 2.5,
    QuantizationType.Q3_K: 3.35,
    QuantizationType.Q4_0: 4.5,
    QuantizationType.Q4_K: 4.5,
    QuantizationType.Q4_K_M: 4.85,
    QuantizationType.Q4_K_S: 4.5,
    QuantizationType.Q5_0: 5.5,
    QuantizationType.Q5_K: 5.5,
    QuantizationType.Q5_K_M: 5.67,
    QuantizationType.Q6_K: 6.57,
    QuantizationType.Q8_0: 8.5,
    QuantizationType.FP16: 16.0,
    QuantizationType.BF16: 16.0,
    QuantizationType.UNKNOWN: 5.0,  # Conservative unknown estimate
}

# KV cache bytes per token per layer (float16 representation)
# Formula: 2 * num_heads * head_dim * sizeof(float16)
# We use a simplified approximation based on model size
KV_CACHE_BYTES_PER_TOKEN_PER_LAYER: dict[float, float] = {
    1.0: 0.0625,  # ~64 MB for 2k context
    3.0: 0.125,
    7.0: 0.25,
    8.0: 0.25,
    9.0: 0.3125,
    13.0: 0.375,
    14.0: 0.375,
    30.0: 0.5,
    70.0: 0.75,
}

# GPU memory overhead: Ollama runtime + CUDA context (GB)
RUNTIME_OVERHEAD_GB = 0.5

# Approximate layer counts by parameter size
PARAM_TO_LAYERS: dict[float, int] = {
    1.0: 22,
    3.0: 26,
    7.0: 32,
    8.0: 32,
    9.0: 42,
    13.0: 40,
    14.0: 40,
    30.0: 60,
    70.0: 80,
}


def _get_nearest(table: dict[float, float | int], key: float) -> float | int:
    """Return the value from a float-keyed dict closest to the given key.

    Args:
        table: Dictionary with float keys.
        key: Lookup value.

    Returns:
        The value whose key is nearest to `key`.
    """
    if not table:
        return 0.0
    nearest_key = min(table.keys(), key=lambda k: abs(k - key))
    return table[nearest_key]


def compute_base_vram(model: OllamaModel) -> float:
    """Estimate GPU VRAM needed for model weights only (no KV cache).

    Uses the bits-per-weight formula:
        VRAM_GB = (params_B * 1e9 * bpw) / (8 * 1024^3)

    Args:
        model: OllamaModel to estimate.

    Returns:
        Estimated base VRAM in GB.
    """
    if model.param_size_b <= 0:
        # Fall back to disk size as a rough proxy
        return model.disk_size_gb * 1.05

    bpw = BITS_PER_WEIGHT.get(model.quantization, 5.0)
    # Convert: params * bits / bits_per_byte / bytes_per_gb
    vram_gb = (model.param_size_b * 1e9 * bpw) / (8 * 1024**3)
    # Add runtime overhead
    return round(vram_gb + RUNTIME_OVERHEAD_GB, 2)


def compute_kv_cache_vram(model: OllamaModel, context_length: int) -> float:
    """Estimate VRAM consumed by KV cache at a given context length.

    Formula approximation:
        KV = bytes_per_token_per_layer * num_layers * context_len / GB

    Args:
        model: Target OllamaModel.
        context_length: Context window size in tokens.

    Returns:
        KV cache VRAM in GB.
    """
    param_size = max(model.param_size_b, 1.0)
    layers = int(_get_nearest(PARAM_TO_LAYERS, param_size))
    bytes_per_tok_per_layer = float(
        _get_nearest(KV_CACHE_BYTES_PER_TOKEN_PER_LAYER, param_size)
    )

    # Total bytes for full context, both K and V
    total_bytes = bytes_per_tok_per_layer * layers * context_length * 2
    kv_gb = total_bytes / 1024  # bytes_per_tok_per_layer already in MB scale
    return round(kv_gb / 1024, 3)


def estimate_vram(
    model: OllamaModel,
    context_length: int,
    available_vram_gb: float,
) -> VRAMEstimate:
    """Compute full VRAM estimate and determine GPU fit status.

    Args:
        model: OllamaModel to analyze.
        context_length: Target context window in tokens.
        available_vram_gb: Total VRAM on the GPU (0 if CPU-only).

    Returns:
        VRAMEstimate with fit status, offload info, and totals.
    """
    base = compute_base_vram(model)
    kv = compute_kv_cache_vram(model, context_length)
    total = round(base + kv, 2)

    param_size = max(model.param_size_b, 1.0)
    total_layers = int(_get_nearest(PARAM_TO_LAYERS, param_size))

    if available_vram_gb <= 0:
        # CPU-only system
        fit_status = GPUFitStatus.OOM
        offload_layers = total_layers
        cpu_fallback = True
    elif total <= available_vram_gb * 0.95:
        # Comfortably fits (leaving 5% headroom)
        fit_status = GPUFitStatus.FITS
        offload_layers = 0
        cpu_fallback = False
    elif base <= available_vram_gb * 0.95:
        # Model weights fit but KV cache may overflow — partial offload
        fit_status = GPUFitStatus.PARTIAL
        # Estimate how many layers can fit
        vram_per_layer = base / total_layers if total_layers > 0 else base
        layers_in_gpu = int(available_vram_gb / vram_per_layer)
        offload_layers = max(0, total_layers - layers_in_gpu)
        cpu_fallback = False
    else:
        # Even weights don't fit
        fit_status = GPUFitStatus.OOM
        offload_layers = total_layers
        cpu_fallback = True

    return VRAMEstimate(
        model_name=model.name,
        context_length=context_length,
        base_vram_gb=base,
        kv_cache_gb=kv,
        total_vram_gb=total,
        fit_status=fit_status,
        offload_layers=offload_layers,
        total_layers=total_layers,
        cpu_fallback=cpu_fallback,
    )


def simulate_multi_model_load(
    models: list[OllamaModel],
    context_length: int,
    system_info: SystemInfo,
) -> MultiModelSimulation:
    """Simulate loading multiple models simultaneously into VRAM.

    This is useful for scenarios where you want to hot-swap between models
    or run concurrent inference with model caching enabled.

    Args:
        models: List of models to load together.
        context_length: Context window for each model.
        system_info: Hardware information including VRAM.

    Returns:
        MultiModelSimulation with fit analysis and recommendations.
    """
    available_vram = (
        system_info.gpu.vram_total_gb if system_info.gpu.is_available else 0.0
    )
    model_names = [m.name for m in models]
    total_vram = 0.0
    recommendations: list[str] = []

    for model in models:
        est = estimate_vram(model, context_length, available_vram)
        total_vram += est.total_vram_gb

    fits = total_vram <= available_vram * 0.90  # 10% safety margin

    if not fits:
        overflow = total_vram - available_vram
        recommendations.append(
            f"Total VRAM needed {total_vram:.1f} GB exceeds available "
            f"{available_vram:.1f} GB by {overflow:.1f} GB."
        )
        recommendations.append(
            "Consider using lower quantization (Q4 instead of Q8) to reduce memory."
        )
        if len(models) > 2:
            recommendations.append(
                "Reduce concurrent models to 2 or use sequential loading."
            )
    else:
        recommendations.append(
            f"All {len(models)} models fit in {available_vram:.1f} GB VRAM "
            f"({total_vram:.1f} GB used)."
        )

    return MultiModelSimulation(
        model_names=model_names,
        total_vram_needed_gb=round(total_vram, 2),
        fits_in_gpu=fits,
        recommendations=recommendations,
    )


def compute_kv_scaling_curve(
    model: OllamaModel,
    available_vram_gb: float,
    context_lengths: Optional[list[int]] = None,
) -> list[tuple[int, VRAMEstimate]]:
    """Generate VRAM estimates across multiple context lengths.

    Useful for visualizing how VRAM usage scales with context size.

    Args:
        model: Target OllamaModel.
        available_vram_gb: Available VRAM for fit determination.
        context_lengths: List of context sizes to evaluate.
            Defaults to [512, 1024, 2048, 4096, 8192, 16384, 32768].

    Returns:
        List of (context_length, VRAMEstimate) tuples sorted by context length.
    """
    from typing import Optional  # local import to avoid circular

    if context_lengths is None:
        context_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    results = []
    for ctx in sorted(context_lengths):
        est = estimate_vram(model, ctx, available_vram_gb)
        results.append((ctx, est))
    return results
