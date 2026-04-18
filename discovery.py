"""
Ollama model discovery module.

Parses the output of `ollama list` and enriches each model with metadata
such as quantization level, parameter size, GGUF status, and capability tags.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import Optional

from models import ModelCapability, OllamaModel, QuantizationType
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Static capability mapping keyed by partial model name (lowercase)
# ---------------------------------------------------------------------------
CAPABILITY_MAP: dict[str, list[ModelCapability]] = {
    "llama": [
        ModelCapability.CHAT,
        ModelCapability.INSTRUCTION,
        ModelCapability.REASONING,
    ],
    "mistral": [
        ModelCapability.CHAT,
        ModelCapability.INSTRUCTION,
        ModelCapability.CODING,
    ],
    "codellama": [ModelCapability.CODING, ModelCapability.INSTRUCTION],
    "deepseek": [ModelCapability.REASONING, ModelCapability.CODING],
    "gemma": [
        ModelCapability.CHAT,
        ModelCapability.INSTRUCTION,
        ModelCapability.MULTILINGUAL,
    ],
    "phi": [ModelCapability.REASONING, ModelCapability.CODING],
    "qwen": [
        ModelCapability.MULTILINGUAL,
        ModelCapability.CODING,
        ModelCapability.CHAT,
    ],
    "nomic": [ModelCapability.EMBEDDING],
    "mxbai": [ModelCapability.EMBEDDING],
    "llava": [ModelCapability.VISION, ModelCapability.CHAT],
    "bakllava": [ModelCapability.VISION, ModelCapability.CHAT],
    "wizard": [ModelCapability.REASONING, ModelCapability.CODING],
    "neural": [ModelCapability.CHAT, ModelCapability.INSTRUCTION],
    "orca": [ModelCapability.REASONING, ModelCapability.INSTRUCTION],
    "vicuna": [ModelCapability.CHAT, ModelCapability.INSTRUCTION],
    "solar": [ModelCapability.CHAT, ModelCapability.REASONING],
    "yi": [ModelCapability.MULTILINGUAL, ModelCapability.CHAT],
    "stablelm": [ModelCapability.CODING, ModelCapability.CHAT],
    "tinyllama": [ModelCapability.CHAT],
    "falcon": [ModelCapability.CHAT, ModelCapability.INSTRUCTION],
}

# Approximate layer counts by parameter size (used for VRAM offload estimation)
PARAM_TO_LAYERS: dict[float, int] = {
    1.0: 22,
    3.0: 26,
    7.0: 32,
    8.0: 32,
    9.0: 42,
    13.0: 40,
    14.0: 40,
    30.0: 60,
    34.0: 60,
    70.0: 80,
}


def _parse_size_string(size_str: str) -> float:
    """Convert a human-readable size string to gigabytes.

    Args:
        size_str: String like '4.2 GB', '512 MB', '1.1 TB'.

    Returns:
        Size in gigabytes as float. Returns 0.0 on parse failure.
    """
    size_str = size_str.strip().upper()
    # Match patterns like "4.2 GB", "512MB", "1.1 TB"
    match = re.match(r"([\d.]+)\s*(GB|MB|KB|TB|GIB|MIB|KIB|TIB)", size_str)
    if not match:
        logger.warning(f"Failed to parse size string: '{size_str}'")
        return 0.0
    value = float(match.group(1))
    unit = match.group(2)
    unit_map = {
        "KB": 1e-6,
        "KIB": 1e-6,
        "MB": 1e-3,
        "MIB": 1e-3,
        "GB": 1.0,
        "GIB": 1.0,
        "TB": 1000.0,
        "TIB": 1000.0,
    }
    return value * unit_map.get(unit, 1.0)


def _parse_param_size(name: str) -> float:
    """Extract parameter count in billions from model name.

    Args:
        name: Model name string, e.g. 'llama3:8b', 'mistral:7b-q4_k_m'.

    Returns:
        Parameter count as float billions. 0.0 if not found.

    Examples:
        >>> _parse_param_size("llama3:8b")
        8.0
        >>> _parse_param_size("phi3:3.8b")
        3.8
    """
    # Pattern: digits optionally followed by decimal, then 'b' (case-insensitive)
    match = re.search(r"(\d+(?:\.\d+)?)\s*b(?:[^a-z]|$)", name.lower())
    if match:
        return float(match.group(1))
    return 0.0


def _parse_quantization(name: str) -> QuantizationType:
    """Identify quantization type from model name.

    Args:
        name: Full model name including tag (e.g., 'llama3:8b-q4_k_m').

    Returns:
        QuantizationType enum value.
    """
    name_lower = name.lower()
    # Check from most specific to least specific
    quant_patterns = [
        (r"q4[_-]?k[_-]?m", QuantizationType.Q4_K_M),
        (r"q4[_-]?k[_-]?s", QuantizationType.Q4_K_S),
        (r"q4[_-]?k", QuantizationType.Q4_K),
        (r"q4[_-]?0", QuantizationType.Q4_0),
        (r"q4", QuantizationType.Q4_0),
        (r"q5[_-]?k[_-]?m", QuantizationType.Q5_K_M),
        (r"q5[_-]?k", QuantizationType.Q5_K),
        (r"q5[_-]?0", QuantizationType.Q5_0),
        (r"q5", QuantizationType.Q5_0),
        (r"q6[_-]?k", QuantizationType.Q6_K),
        (r"q8[_-]?0", QuantizationType.Q8_0),
        (r"q8", QuantizationType.Q8_0),
        (r"q3[_-]?k", QuantizationType.Q3_K),
        (r"q2[_-]?k", QuantizationType.Q2_K),
        (r"fp16", QuantizationType.FP16),
        (r"bf16", QuantizationType.BF16),
        (r"f16", QuantizationType.FP16),
    ]
    for pattern, quant_type in quant_patterns:
        if re.search(pattern, name_lower):
            return quant_type
    # Default: non-quantized models are likely FP16
    return QuantizationType.UNKNOWN


def _resolve_capabilities(name: str) -> list[ModelCapability]:
    """Determine capabilities based on model family name.

    Args:
        name: Full model name string.

    Returns:
        List of ModelCapability tags.
    """
    name_lower = name.lower()
    for family_key, caps in CAPABILITY_MAP.items():
        if family_key in name_lower:
            return caps
    # Generic fallback — assume it can at least chat
    return [ModelCapability.CHAT]


def _resolve_family(name: str) -> str:
    """Extract the model family name from the full identifier.

    Args:
        name: Full model name string.

    Returns:
        Lowercase family string, e.g. 'llama', 'mistral'.
    """
    base = name.split(":")[0].lower()
    # Strip numeric suffixes (e.g., 'llama3' -> 'llama')
    return re.sub(r"\d+$", "", base)


def _estimate_context_length(name: str, param_size: float) -> int:
    """Estimate default context window by model name and size.

    Args:
        name: Full model name.
        param_size: Parameter count in billions.

    Returns:
        Estimated context length in tokens.
    """
    name_lower = name.lower()
    # Some models are known for large contexts
    if "llama3" in name_lower:
        return 8192
    if "gemma2" in name_lower:
        return 8192
    if "qwen" in name_lower:
        return 32768
    if "phi3" in name_lower:
        return 128000
    if "mistral" in name_lower:
        return 32768
    if "deepseek" in name_lower:
        return 32768
    # Default by size: larger models often have bigger contexts
    if param_size >= 30:
        return 4096
    return 4096


def _get_layer_count(param_size_b: float) -> int:
    """Estimate transformer layer count from parameter size.

    Args:
        param_size_b: Parameter count in billions.

    Returns:
        Estimated number of transformer layers.
    """
    # Find closest matching param size
    closest = min(PARAM_TO_LAYERS.keys(), key=lambda k: abs(k - param_size_b))
    return PARAM_TO_LAYERS[closest]


def parse_ollama_list_output(raw_output: str) -> list[OllamaModel]:
    """Parse raw text output from `ollama list` into OllamaModel instances.

    The expected format is:
        NAME               ID            SIZE      MODIFIED
        llama3:8b          ...           4.7 GB    2 weeks ago

    Args:
        raw_output: Raw stdout string from `ollama list`.

    Returns:
        List of parsed OllamaModel instances. Empty list if no models found.
    """
    models: list[OllamaModel] = []
    lines = raw_output.strip().splitlines()

    if not lines:
        return models

    # Skip the header line (NAME  ID  SIZE  MODIFIED)
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            logger.warning(f"Skipping malformed line: '{line}'")
            continue

        name = parts[0]
        # Size field: join parts that look like "4.7 GB" into one token
        # The SIZE column is at index 2, unit at index 3
        size_str = ""
        try:
            # Ollama list format: NAME  ID  SIZE  UNIT  MODIFIED...
            if len(parts) >= 4 and parts[3].upper() in ("GB", "MB", "KB", "TB"):
                size_str = f"{parts[2]} {parts[3]}"
            elif len(parts) >= 3:
                size_str = parts[2]
        except IndexError:
            size_str = "0 GB"

        # Parse "2 weeks ago" or ISO datetime for modified_at
        modified_parts = parts[4:] if len(parts) > 4 else []
        modified_at = " ".join(modified_parts)

        param_size = _parse_param_size(name)
        quant = _parse_quantization(name)
        capabilities = _resolve_capabilities(name)
        family = _resolve_family(name)
        disk_size_gb = _parse_size_string(size_str)
        context_len = _estimate_context_length(name, param_size)

        try:
            model = OllamaModel(
                name=name,
                display_name=name,
                param_size_b=param_size,
                quantization=quant,
                disk_size_gb=disk_size_gb,
                is_gguf=True,  # Ollama models are GGUF by default
                context_length=context_len,
                capabilities=capabilities,
                family=family,
                modified_at=modified_at,
            )
            models.append(model)
        except Exception as e:
            logger.error(f"Failed to create OllamaModel for '{name}': {e}")
            continue

    return models


def fetch_models() -> list[OllamaModel]:
    """Run `ollama list` and return parsed model objects.

    Returns:
        List of OllamaModel instances. Empty if Ollama is not available.

    Raises:
        RuntimeError: If `ollama` binary is not found on PATH.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            logger.error(f"ollama list failed: {result.stderr}")
            return []
        return parse_ollama_list_output(result.stdout)

    except FileNotFoundError:
        logger.error("'ollama' binary not found. Is Ollama installed?")
        raise RuntimeError(
            "Ollama not found. Please install it from https://ollama.com"
        )
    except subprocess.TimeoutExpired:
        logger.error("ollama list timed out after 15 seconds")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching models: {e}", exc_info=True)
        return []


def get_model_show(model_name: str) -> dict:
    """Run `ollama show --modelfile <name>` to get detailed model info.

    Args:
        model_name: The model identifier to inspect.

    Returns:
        Dictionary with 'modelfile' key. Empty dict on failure.
    """
    try:
        result = subprocess.run(
            ["ollama", "show", "--modelfile", model_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return {"modelfile": result.stdout}
        logger.warning(f"ollama show failed for '{model_name}': {result.stderr}")
        return {}
    except Exception as e:
        logger.error(f"Error running ollama show for '{model_name}': {e}")
        return {}
