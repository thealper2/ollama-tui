"""
System hardware scanner for Ollama TUI.

Detects GPU (via nvidia-smi), RAM (/proc/meminfo or psutil),
and CPU information to provide accurate VRAM fit estimates.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from typing import Optional

from models import GPUInfo, SystemInfo
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _run_command(cmd: list[str], timeout: int = 5) -> Optional[str]:
    """Run a shell command and return stdout as string.

    Args:
        cmd: Command and arguments list.
        timeout: Max seconds to wait.

    Returns:
        stdout string or None on failure.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.debug(f"Command {cmd} exited {result.returncode}: {result.stderr}")
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.debug(f"Command {cmd} failed: {e}")
        return None


def _scan_nvidia_gpu() -> Optional[GPUInfo]:
    """Query nvidia-smi for GPU metadata.

    Returns:
        GPUInfo populated from nvidia-smi, or None if no NVIDIA GPU found.
    """
    # Query CSV format for easy parsing
    query = "name,memory.total,memory.free,driver_version,compute_cap"
    output = _run_command(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None

    # Take the first GPU line if multiple GPUs present
    first_line = output.splitlines()[0]
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) < 4:
        logger.warning(f"Unexpected nvidia-smi output format: '{first_line}'")
        return None

    try:
        name = parts[0]
        # nvidia-smi returns MiB; convert to GB
        vram_total_gb = float(parts[1]) / 1024
        vram_free_gb = float(parts[2]) / 1024
        driver_version = parts[3] if len(parts) > 3 else ""
        compute_cap = parts[4] if len(parts) > 4 else ""

        return GPUInfo(
            name=name,
            vram_total_gb=round(vram_total_gb, 2),
            vram_free_gb=round(vram_free_gb, 2),
            compute_capability=compute_cap,
            driver_version=driver_version,
            is_available=True,
        )
    except (ValueError, IndexError) as e:
        logger.error(f"Failed to parse nvidia-smi output: {e}")
        return None


def _scan_amd_gpu() -> Optional[GPUInfo]:
    """Query rocm-smi for AMD GPU metadata.

    Returns:
        GPUInfo if AMD GPU is available, None otherwise.
    """
    output = _run_command(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    if not output:
        return None

    # Parse rocm-smi CSV — format varies by version, use heuristic
    lines = [
        l for l in output.splitlines() if "vram" in l.lower() or "total" in l.lower()
    ]
    total_mb = 0.0
    free_mb = 0.0

    for line in output.splitlines():
        m_total = re.search(r"VRAM Total Memory.*?(\d+)", line, re.IGNORECASE)
        m_free = re.search(r"VRAM Free Memory.*?(\d+)", line, re.IGNORECASE)
        if m_total:
            total_mb = float(m_total.group(1))
        if m_free:
            free_mb = float(m_free.group(1))

    if total_mb > 0:
        return GPUInfo(
            name="AMD GPU (rocm-smi)",
            vram_total_gb=round(total_mb / 1024, 2),
            vram_free_gb=round(free_mb / 1024, 2),
            is_available=True,
        )
    return None


def _scan_apple_silicon() -> Optional[GPUInfo]:
    """Detect Apple Silicon shared memory as effective VRAM.

    On Apple M-series, RAM is shared with GPU. We use total RAM
    and assume ~75% is available for model loading.

    Returns:
        GPUInfo representing Apple unified memory, or None.
    """
    if platform.system() != "Darwin":
        return None

    # Check for Apple Silicon via sysctl
    cpu_brand = _run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
    hw_model = _run_command(["sysctl", "-n", "hw.model"])

    is_apple_silicon = (cpu_brand and "apple" in cpu_brand.lower()) or (
        hw_model and "mac" in hw_model.lower() and cpu_brand is None
    )

    if not is_apple_silicon:
        # Fallback: check for 'Apple M' in hardware overview
        sp_output = _run_command(["system_profiler", "SPHardwareDataType"])
        is_apple_silicon = sp_output and "Apple M" in sp_output

    if not is_apple_silicon:
        return None

    # Get total RAM via sysctl
    mem_bytes_str = _run_command(["sysctl", "-n", "hw.memsize"])
    total_gb = 0.0
    if mem_bytes_str:
        try:
            total_gb = int(mem_bytes_str) / (1024**3)
        except ValueError:
            pass

    # Unified memory: assume 75% is GPU-usable
    vram_total = round(total_gb * 0.75, 1)
    vram_free = round(vram_total * 0.8, 1)

    # Extract chip name
    chip_name = "Apple Silicon"
    if sp_output := _run_command(["system_profiler", "SPHardwareDataType"]):
        m = re.search(r"Chip:\s*(.+)", sp_output)
        if m:
            chip_name = m.group(1).strip()

    return GPUInfo(
        name=chip_name,
        vram_total_gb=vram_total,
        vram_free_gb=vram_free,
        compute_capability="metal",
        is_available=True,
    )


def _scan_ram_linux() -> tuple[float, float]:
    """Read total and free RAM from /proc/meminfo on Linux.

    Returns:
        Tuple of (total_gb, free_gb). (0, 0) on failure.
    """
    try:
        with open("/proc/meminfo", "r") as f:
            content = f.read()
        total_kb = free_kb = 0
        for line in content.splitlines():
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                # MemAvailable is a better metric than MemFree
                free_kb = int(line.split()[1])
        return round(total_kb / 1024 / 1024, 1), round(free_kb / 1024 / 1024, 1)
    except Exception as e:
        logger.warning(f"Failed to read /proc/meminfo: {e}")
        return 0.0, 0.0


def _scan_ram_cross_platform() -> tuple[float, float]:
    """Get RAM info using psutil if available, falling back to platform tools.

    Returns:
        Tuple of (total_gb, free_gb).
    """
    try:
        import psutil

        vm = psutil.virtual_memory()
        total = round(vm.total / 1024**3, 1)
        free = round(vm.available / 1024**3, 1)
        return total, free
    except ImportError:
        pass

    # Linux-specific fallback
    if platform.system() == "Linux":
        return _scan_ram_linux()

    # macOS fallback via sysctl + vm_stat
    if platform.system() == "Darwin":
        mem_str = _run_command(["sysctl", "-n", "hw.memsize"])
        if mem_str:
            try:
                total = int(mem_str) / 1024**3
                return round(total, 1), round(total * 0.5, 1)  # Estimate free as 50%
            except ValueError:
                pass

    return 0.0, 0.0


def _scan_cpu() -> tuple[int, str]:
    """Detect CPU thread count and model name.

    Returns:
        Tuple of (thread_count, model_name).
    """
    threads = os.cpu_count() or 1

    # Try to get a meaningful CPU name
    cpu_name = "Unknown CPU"
    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line.lower():
                        cpu_name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    elif system == "Darwin":
        brand = _run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            cpu_name = brand
    elif system == "Windows":
        output = _run_command(["wmic", "cpu", "get", "name", "/value"])
        if output:
            m = re.search(r"Name=(.+)", output)
            if m:
                cpu_name = m.group(1).strip()

    return threads, cpu_name


def scan_system() -> SystemInfo:
    """Perform a full hardware scan and return a SystemInfo snapshot.

    Attempts GPU detection in order: NVIDIA → AMD → Apple Silicon.
    Collects RAM and CPU information from OS-specific sources.

    Returns:
        SystemInfo with all discovered hardware details populated.
    """
    logger.info("Starting system hardware scan")

    # --- GPU Detection ---
    gpu_info = _scan_nvidia_gpu()
    if gpu_info is None:
        gpu_info = _scan_amd_gpu()
    if gpu_info is None:
        gpu_info = _scan_apple_silicon()
    if gpu_info is None:
        # No GPU found — create a placeholder with is_available=False
        gpu_info = GPUInfo(
            name="No GPU detected",
            vram_total_gb=0.0,
            vram_free_gb=0.0,
            is_available=False,
        )
        logger.info("No GPU detected; will use CPU-only estimates")
    else:
        logger.info(
            f"GPU detected: {gpu_info.name} | VRAM: {gpu_info.vram_total_gb:.1f} GB"
        )

    # --- RAM Detection ---
    ram_total, ram_free = _scan_ram_cross_platform()
    logger.info(f"RAM: {ram_total:.1f} GB total, {ram_free:.1f} GB free")

    # --- CPU Detection ---
    cpu_threads, cpu_model = _scan_cpu()
    logger.info(f"CPU: {cpu_model} ({cpu_threads} threads)")

    return SystemInfo(
        gpu=gpu_info,
        ram_total_gb=ram_total,
        ram_free_gb=ram_free,
        cpu_threads=cpu_threads,
        cpu_model=cpu_model,
        os_name=platform.system(),
    )
