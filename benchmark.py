"""
Benchmark runner for Ollama TUI.

Runs real inference benchmarks against the local Ollama API endpoint
and measures tokens/sec, first-token latency, and total latency.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional

from models import BenchmarkPreset, BenchmarkResult
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Ollama API configuration
# ---------------------------------------------------------------------------
OLLAMA_API_BASE = "http://localhost:11434"
GENERATE_ENDPOINT = f"{OLLAMA_API_BASE}/api/generate"
DEFAULT_TIMEOUT = 120  # seconds

# ---------------------------------------------------------------------------
# Prompt presets for each benchmark type
# ---------------------------------------------------------------------------
BENCHMARK_PROMPTS: dict[BenchmarkPreset, str] = {
    BenchmarkPreset.SHORT: ("What is the capital of France? Answer in one sentence."),
    BenchmarkPreset.LONG: (
        "Write a detailed 5-paragraph essay comparing the philosophical differences "
        "between rationalism and empiricism. Include key thinkers for each tradition "
        "and discuss how each approach has influenced modern science and technology."
    ),
    BenchmarkPreset.REASONING: (
        "You have 12 balls, all identical in appearance, but one is either heavier "
        "or lighter than the others. Using a balance scale in exactly 3 weighings, "
        "how can you identify the odd ball AND determine if it is heavier or lighter? "
        "Walk through your reasoning step-by-step."
    ),
    BenchmarkPreset.CODING: (
        "Write a Python function that implements a binary search tree with insert, "
        "search, and in-order traversal methods. Include docstrings and type hints. "
        "Then write unit tests for all three methods."
    ),
}


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    """Make a POST request with JSON body and return parsed response.

    Args:
        url: Target URL.
        payload: Request body as dict (will be JSON-encoded).
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response dict.

    Raises:
        ConnectionRefusedError: If Ollama server is not running.
        urllib.error.URLError: On network errors.
        json.JSONDecodeError: If response is not valid JSON.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except ConnectionRefusedError:
        raise ConnectionRefusedError(
            "Ollama server is not running. Start it with: ollama serve"
        )


def _stream_generate(
    model_name: str,
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[float, float, int]:
    """Stream a generation request and measure performance metrics.

    Connects to the Ollama streaming API and reads chunks one by one,
    recording the time to first token and total generation time.

    Args:
        model_name: Ollama model identifier.
        prompt: Input prompt text.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (first_token_ms, total_ms, total_tokens).

    Raises:
        ConnectionRefusedError: If Ollama server is unreachable.
        RuntimeError: On API errors or invalid responses.
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": 256,  # Cap output for fair comparison
            "temperature": 0.1,  # Low temperature for reproducibility
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        GENERATE_ENDPOINT,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    first_token_ms = 0.0
    total_tokens = 0
    request_start = time.perf_counter()
    first_token_received = False

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not first_token_received and chunk.get("response"):
                    first_token_ms = (time.perf_counter() - request_start) * 1000
                    first_token_received = True
                    logger.debug(f"First token latency: {first_token_ms:.1f} ms")

                total_tokens += 1

                # Final chunk contains eval_count (actual token count from model)
                if chunk.get("done"):
                    actual_tokens = chunk.get("eval_count", total_tokens)
                    total_tokens = actual_tokens
                    break

    except ConnectionRefusedError:
        raise ConnectionRefusedError(
            "Ollama server is not running. Start it with: ollama serve"
        )
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error connecting to Ollama: {e}")

    total_ms = (time.perf_counter() - request_start) * 1000
    return first_token_ms, total_ms, total_tokens


def run_benchmark(
    model_name: str,
    preset: BenchmarkPreset,
    warmup: bool = True,
) -> BenchmarkResult:
    """Run a complete benchmark for a single model/preset combination.

    Optionally runs a short warmup inference first to pre-load the model
    into VRAM before measuring, giving more accurate throughput numbers.

    Args:
        model_name: Ollama model identifier (e.g., 'llama3:8b').
        preset: Which benchmark prompt to use.
        warmup: If True, run a short warmup request first (recommended).

    Returns:
        BenchmarkResult with measured performance metrics.
    """
    logger.info(f"Starting benchmark: model={model_name}, preset={preset.value}")

    prompt = BENCHMARK_PROMPTS[preset]

    try:
        # Warmup pass: load model into GPU before timing
        if warmup:
            logger.debug(f"Running warmup for {model_name}")
            try:
                warmup_payload = {
                    "model": model_name,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {"num_predict": 5},
                }
                _post_json(GENERATE_ENDPOINT, warmup_payload, timeout=60)
            except Exception as e:
                logger.warning(f"Warmup failed (continuing anyway): {e}")

        # Main benchmark run
        first_token_ms, total_ms, total_tokens = _stream_generate(
            model_name, prompt, timeout=DEFAULT_TIMEOUT
        )

        # Compute tokens/sec — exclude first-token latency for throughput
        # Use total time as denominator for conservative measurement
        tps = (total_tokens / (total_ms / 1000)) if total_ms > 0 else 0.0

        result = BenchmarkResult(
            model_name=model_name,
            preset=preset,
            tokens_per_sec=round(tps, 1),
            first_token_latency_ms=round(first_token_ms, 1),
            total_latency_ms=round(total_ms, 1),
            total_tokens=total_tokens,
        )
        logger.info(
            f"Benchmark complete: {model_name} | "
            f"{tps:.1f} tok/s | {first_token_ms:.0f}ms TTFT | "
            f"score={result.score_label}"
        )
        return result

    except ConnectionRefusedError as e:
        logger.error(f"Connection error during benchmark: {e}")
        return BenchmarkResult(
            model_name=model_name,
            preset=preset,
            error=str(e),
        )
    except Exception as e:
        logger.error(f"Benchmark failed for '{model_name}': {e}", exc_info=True)
        return BenchmarkResult(
            model_name=model_name,
            preset=preset,
            error=str(e),
        )


def check_ollama_running() -> bool:
    """Verify that the Ollama API server is reachable.

    Returns:
        True if the server responds, False otherwise.
    """
    try:
        req = urllib.request.Request(f"{OLLAMA_API_BASE}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False
