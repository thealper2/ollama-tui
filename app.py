"""
Main TUI application class for Ollama TUI.

Orchestrates all panels, user interactions, and navigation using the
Rich library for rendering. Implements an event-driven interactive loop.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.style import Style
from rich.table import Table
from rich.text import Text

from benchmark import check_ollama_running, run_benchmark
from discovery import fetch_models
from models import (
    BenchmarkPreset,
    BenchmarkResult,
    GPUFitStatus,
    OllamaModel,
    QuantizationType,
    SystemInfo,
    VRAMEstimate,
)
from recommender import (
    analyze_quantization_impact,
    estimate_prompt_cost,
    estimate_rag_pipeline,
    recommend_models,
)
from scanner import scan_system
from utils.logger import setup_logger
from vram_calculator import (
    compute_kv_scaling_curve,
    estimate_vram,
    simulate_multi_model_load,
)

logger = setup_logger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------
APP_TITLE = "🧠 Ollama TUI"
APP_VERSION = "1.0.0"

# Color palette
COLOR_OK = "bold green"
COLOR_WARN = "bold yellow"
COLOR_ERR = "bold red"
COLOR_INFO = "bold cyan"
COLOR_DIM = "dim"
COLOR_ACCENT = "bold magenta"

FIT_ICONS = {
    GPUFitStatus.FITS: ("✅", "green"),
    GPUFitStatus.PARTIAL: ("⚠", "yellow"),
    GPUFitStatus.OOM: ("❌", "red"),
}

QUANT_COLORS: dict[str, str] = {
    "q2": "red",
    "q3": "orange3",
    "q4": "yellow",
    "q5": "green",
    "q6": "cyan",
    "q8": "bright_cyan",
    "fp16": "bright_white",
    "bf16": "bright_white",
    "unknown": "dim white",
}


def _quant_color(quant: QuantizationType) -> str:
    """Return Rich color string for a quantization type."""
    key = quant.value.lower()[:3]
    return QUANT_COLORS.get(key, "white")


def _header_panel() -> Panel:
    """Render the application header with title and version."""
    header = Text(justify="center")
    header.append(f" {APP_TITLE} ", style="bold magenta on black")
    header.append(f"  v{APP_VERSION} ", style="dim")
    return Panel(header, style="magenta", padding=(0, 1))


def _footer_help(options: list[tuple[str, str]]) -> Text:
    """Build a compact footer help line from (key, description) pairs.

    Args:
        options: List of (key_string, description) tuples.

    Returns:
        Formatted Rich Text object.
    """
    text = Text()
    for i, (key, desc) in enumerate(options):
        if i > 0:
            text.append("  │  ", style="dim")
        text.append(f"[{key}]", style="bold cyan")
        text.append(f" {desc}", style="dim")
    return text


class OllamaTUIApp:
    """Main application controller for the Ollama TUI.

    Manages application state, renders all views using Rich,
    and handles user navigation through an interactive menu system.
    """

    def __init__(self) -> None:
        """Initialize app state — models and system info loaded lazily."""
        self.console = console
        self.models: list[OllamaModel] = []
        self.system_info: Optional[SystemInfo] = None
        self.benchmark_results: dict[str, BenchmarkResult] = {}
        self._ollama_running: bool = False

    def _load_data(self) -> None:
        """Load models and system info with Rich progress spinners."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=self.console,
        ) as progress:
            task1 = progress.add_task("Scanning system hardware...", total=None)
            try:
                self.system_info = scan_system()
            except Exception as e:
                logger.error(f"System scan failed: {e}")
                self.system_info = SystemInfo()
            progress.remove_task(task1)

            task2 = progress.add_task("Fetching Ollama models...", total=None)
            try:
                self.models = fetch_models()
                self._ollama_running = check_ollama_running()
            except RuntimeError as e:
                self.console.print(f"\n[bold red]Error:[/bold red] {e}")
                self.models = []
            except Exception as e:
                logger.error(f"Model fetch failed: {e}")
                self.models = []
            progress.remove_task(task2)

    # -----------------------------------------------------------------------
    # View: System Info
    # -----------------------------------------------------------------------

    def _render_system_info(self) -> None:
        """Display a detailed system hardware summary panel."""
        si = self.system_info
        if not si:
            self.console.print("[red]System info not available.[/red]")
            return

        self.console.print()
        self.console.print(Rule("[bold cyan]⚙️  System Information[/bold cyan]"))
        self.console.print()

        # GPU panel
        gpu = si.gpu
        gpu_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        gpu_table.add_column("Key", style="bold cyan", min_width=22)
        gpu_table.add_column("Value", style="white")

        if gpu.is_available:
            vram_bar = self._vram_bar(gpu.vram_free_gb, gpu.vram_total_gb)
            gpu_table.add_row("GPU", gpu.name)
            gpu_table.add_row("VRAM Total", f"{gpu.vram_total_gb:.1f} GB")
            gpu_table.add_row("VRAM Free", f"{gpu.vram_free_gb:.1f} GB  {vram_bar}")
            if gpu.compute_capability:
                gpu_table.add_row("Compute Cap", gpu.compute_capability)
            if gpu.driver_version:
                gpu_table.add_row("Driver", gpu.driver_version)
        else:
            gpu_table.add_row("GPU", "[red]No GPU detected — CPU mode[/red]")

        # CPU/RAM panel
        sys_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        sys_table.add_column("Key", style="bold cyan", min_width=22)
        sys_table.add_column("Value", style="white")

        ram_bar = self._vram_bar(si.ram_free_gb, si.ram_total_gb)
        sys_table.add_row("CPU", si.cpu_model)
        sys_table.add_row("CPU Threads", str(si.cpu_threads))
        sys_table.add_row("RAM Total", f"{si.ram_total_gb:.1f} GB")
        sys_table.add_row("RAM Available", f"{si.ram_free_gb:.1f} GB  {ram_bar}")
        sys_table.add_row("OS", si.os_name)

        ollama_status = (
            "[green]● Running[/green]"
            if self._ollama_running
            else "[red]● Not running[/red]"
        )
        sys_table.add_row("Ollama Server", ollama_status)

        self.console.print(
            Columns(
                [
                    Panel(
                        gpu_table,
                        title="[bold magenta]GPU[/bold magenta]",
                        border_style="magenta",
                    ),
                    Panel(
                        sys_table,
                        title="[bold blue]System[/bold blue]",
                        border_style="blue",
                    ),
                ]
            )
        )

    def _vram_bar(self, used: float, total: float, width: int = 20) -> str:
        """Create a simple ASCII bar for memory usage visualization.

        Args:
            used: Amount being displayed (free memory in our case).
            total: Maximum amount.
            width: Character width of the bar.

        Returns:
            ASCII progress bar string with Rich markup.
        """
        if total <= 0:
            return ""
        # We show "free" so invert for the fill
        fill_ratio = 1.0 - (used / total)
        fill_ratio = max(0.0, min(1.0, fill_ratio))
        filled = int(fill_ratio * width)
        empty = width - filled

        if fill_ratio > 0.8:
            color = "red"
        elif fill_ratio > 0.5:
            color = "yellow"
        else:
            color = "green"

        bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"
        pct = fill_ratio * 100
        return f"{bar} {pct:.0f}% used"

    # -----------------------------------------------------------------------
    # View: Model List
    # -----------------------------------------------------------------------

    def _render_model_list(self) -> None:
        """Display all installed models in a rich interactive table."""
        if not self.models:
            self.console.print(
                "\n[yellow]No models found. Run 'ollama pull <model>' to install one.[/yellow]"
            )
            return

        self.console.print()
        self.console.print(
            Rule(f"[bold cyan]📦 Installed Models ({len(self.models)})[/bold cyan]")
        )
        self.console.print()

        table = Table(
            box=box.ROUNDED,
            border_style="cyan",
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Model Name", min_width=24, no_wrap=True)
        table.add_column("Params", justify="right", min_width=7)
        table.add_column("Quant", min_width=10)
        table.add_column("Disk", justify="right", min_width=8)
        table.add_column("Context", justify="right", min_width=9)
        table.add_column("VRAM Est.", justify="right", min_width=10)
        table.add_column("GPU Fit", justify="center", min_width=10)
        table.add_column("Capabilities", min_width=20)

        vram_avail = (
            self.system_info.gpu.vram_total_gb
            if (self.system_info and self.system_info.gpu.is_available)
            else 0.0
        )

        for idx, model in enumerate(self.models, start=1):
            est = estimate_vram(model, model.context_length, vram_avail)
            icon, color = FIT_ICONS[est.fit_status]

            param_str = f"{model.param_size_b:.1f}B" if model.param_size_b > 0 else "?"
            disk_str = f"{model.disk_size_gb:.1f} GB" if model.disk_size_gb > 0 else "?"
            ctx_str = f"{model.context_length // 1024}K"
            vram_str = f"{est.total_vram_gb:.1f} GB"
            fit_str = Text(f"{icon} {est.fit_status.value}", style=color)

            quant_color = _quant_color(model.quantization)
            quant_str = Text(model.quantization.value.upper(), style=quant_color)

            caps = ", ".join(c.value for c in model.capabilities[:3])
            if len(model.capabilities) > 3:
                caps += "…"

            table.add_row(
                str(idx),
                f"[bold]{model.name}[/bold]",
                param_str,
                quant_str,
                disk_str,
                ctx_str,
                vram_str,
                fit_str,
                caps,
            )

        self.console.print(table)

    # -----------------------------------------------------------------------
    # View: VRAM Calculator
    # -----------------------------------------------------------------------

    def _render_vram_calculator(self) -> None:
        """Interactive VRAM calculator with context scaling and multi-model sim."""
        if not self.models:
            self.console.print("[yellow]No models available.[/yellow]")
            return

        self.console.print()
        self.console.print(Rule("[bold cyan]🧮 VRAM Calculator[/bold cyan]"))
        self.console.print()

        # Model selection
        self._render_model_list()
        self.console.print()

        try:
            choice = IntPrompt.ask(
                "[cyan]Select model number[/cyan]",
                default=1,
            )
            choice = max(1, min(choice, len(self.models)))
        except (ValueError, KeyboardInterrupt):
            return

        model = self.models[choice - 1]
        vram_avail = (
            self.system_info.gpu.vram_total_gb
            if (self.system_info and self.system_info.gpu.is_available)
            else 0.0
        )

        # Context length selection
        self.console.print(
            "\nContext options: [dim]1=2K  2=4K  3=8K  4=16K  5=32K  6=128K[/dim]"
        )
        ctx_map = {1: 2048, 2: 4096, 3: 8192, 4: 16384, 5: 32768, 6: 131072}
        try:
            ctx_choice = IntPrompt.ask("[cyan]Select context length[/cyan]", default=2)
            ctx_choice = max(1, min(ctx_choice, 6))
        except (ValueError, KeyboardInterrupt):
            return
        context_length = ctx_map[ctx_choice]

        # Calculate estimate
        est = estimate_vram(model, context_length, vram_avail)
        icon, color = FIT_ICONS[est.fit_status]

        result_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        result_table.add_column("Key", style="bold cyan", min_width=24)
        result_table.add_column("Value", style="white")

        result_table.add_row("Model", f"[bold]{model.name}[/bold]")
        result_table.add_row(
            "Parameters",
            f"{model.param_size_b:.1f}B" if model.param_size_b else "Unknown",
        )
        result_table.add_row(
            "Quantization",
            Text(
                model.quantization.value.upper(), style=_quant_color(model.quantization)
            ),
        )
        result_table.add_row("Context Length", f"{context_length:,} tokens")
        result_table.add_row("Base VRAM (weights)", f"{est.base_vram_gb:.2f} GB")
        result_table.add_row("KV Cache VRAM", f"{est.kv_cache_gb:.3f} GB")
        result_table.add_row(
            "Total VRAM Needed", f"[bold]{est.total_vram_gb:.2f} GB[/bold]"
        )
        result_table.add_row(
            "Available VRAM",
            f"{vram_avail:.1f} GB" if vram_avail > 0 else "[dim]CPU only[/dim]",
        )
        result_table.add_row(
            "GPU Fit Status",
            Text(f"{icon} {est.fit_status.value.upper()}", style=color),
        )
        if est.offload_layers > 0:
            result_table.add_row(
                "Offloaded Layers", f"{est.offload_layers}/{est.total_layers} → CPU"
            )
        result_table.add_row(
            "CPU Fallback",
            "[red]Yes[/red]" if est.cpu_fallback else "[green]No[/green]",
        )

        self.console.print(
            Panel(
                result_table,
                title=f"[bold]VRAM Estimate: {model.name}[/bold]",
                border_style=color,
            )
        )

        # KV scaling curve
        self.console.print()
        self.console.print("[bold cyan]KV Cache Scaling Curve:[/bold cyan]")

        curve_table = Table(box=box.SIMPLE, header_style="bold cyan", padding=(0, 1))
        curve_table.add_column("Context", justify="right")
        curve_table.add_column("Base VRAM", justify="right")
        curve_table.add_column("+ KV Cache", justify="right")
        curve_table.add_column("Total", justify="right")
        curve_table.add_column("Fit", justify="center")

        scaling = compute_kv_scaling_curve(model, vram_avail)
        for ctx, curve_est in scaling:
            c_icon, c_color = FIT_ICONS[curve_est.fit_status]
            highlight = "[bold]" if ctx == context_length else ""
            curve_table.add_row(
                f"{highlight}{ctx // 1024}K[/bold]" if highlight else f"{ctx // 1024}K",
                f"{curve_est.base_vram_gb:.2f} GB",
                f"+{curve_est.kv_cache_gb:.3f} GB",
                f"{curve_est.total_vram_gb:.2f} GB",
                Text(f"{c_icon}", style=c_color),
            )

        self.console.print(curve_table)

        # Multi-model simulation
        self.console.print()
        if Confirm.ask("[cyan]Simulate multi-model loading?[/cyan]", default=False):
            self._run_multi_model_simulation()

    def _run_multi_model_simulation(self) -> None:
        """Prompt user to select multiple models and show combined VRAM usage."""
        if len(self.models) < 2:
            self.console.print(
                "[yellow]Need at least 2 models for simulation.[/yellow]"
            )
            return

        self.console.print(
            "\n[dim]Enter comma-separated model numbers (e.g., 1,3,4):[/dim]"
        )
        try:
            raw = Prompt.ask("[cyan]Models to simulate[/cyan]")
            indices = [
                int(x.strip()) - 1 for x in raw.split(",") if x.strip().isdigit()
            ]
            indices = [i for i in indices if 0 <= i < len(self.models)]
        except (ValueError, KeyboardInterrupt):
            return

        if not indices:
            self.console.print("[yellow]No valid selections.[/yellow]")
            return

        selected = [self.models[i] for i in indices]
        sim = simulate_multi_model_load(selected, 4096, self.system_info)

        color = "green" if sim.fits_in_gpu else "red"
        icon = "✅" if sim.fits_in_gpu else "❌"

        sim_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        sim_table.add_column("Key", style="bold cyan", min_width=24)
        sim_table.add_column("Value")

        sim_table.add_row("Models", ", ".join(sim.model_names))
        sim_table.add_row(
            "Total VRAM Needed", f"[bold]{sim.total_vram_needed_gb:.2f} GB[/bold]"
        )
        vram_avail = (
            self.system_info.gpu.vram_total_gb
            if self.system_info.gpu.is_available
            else 0.0
        )
        sim_table.add_row("Available VRAM", f"{vram_avail:.1f} GB")
        sim_table.add_row(
            "Fits Together",
            Text(f"{icon} {'YES' if sim.fits_in_gpu else 'NO'}", style=color),
        )

        self.console.print(
            Panel(
                sim_table,
                title="[bold]Multi-Model Simulation[/bold]",
                border_style=color,
            )
        )

        for rec in sim.recommendations:
            self.console.print(f"  [dim]→[/dim] {rec}")

    # -----------------------------------------------------------------------
    # View: Benchmark
    # -----------------------------------------------------------------------

    def _render_benchmark(self) -> None:
        """Interactive benchmark runner with model and preset selection."""
        if not self.models:
            self.console.print("[yellow]No models available.[/yellow]")
            return

        if not self._ollama_running:
            self.console.print(
                "[red]Ollama server is not running. Start it with: ollama serve[/red]"
            )
            return

        self.console.print()
        self.console.print(Rule("[bold cyan]⚡ Benchmark Runner[/bold cyan]"))
        self.console.print()

        self._render_model_list()
        self.console.print()

        try:
            choice = IntPrompt.ask(
                "[cyan]Select model number to benchmark[/cyan]", default=1
            )
            choice = max(1, min(choice, len(self.models)))
        except (ValueError, KeyboardInterrupt):
            return

        model = self.models[choice - 1]

        # Preset selection
        preset_map = {
            1: BenchmarkPreset.SHORT,
            2: BenchmarkPreset.LONG,
            3: BenchmarkPreset.REASONING,
            4: BenchmarkPreset.CODING,
        }
        self.console.print(
            "\nPresets: [dim]1=Short  2=Long  3=Reasoning  4=Coding[/dim]"
        )
        try:
            preset_choice = IntPrompt.ask(
                "[cyan]Select benchmark preset[/cyan]", default=1
            )
            preset_choice = max(1, min(preset_choice, 4))
        except (ValueError, KeyboardInterrupt):
            return

        preset = preset_map[preset_choice]

        self.console.print(
            f"\n[dim]Running {preset.value} benchmark on [bold]{model.name}[/bold]...[/dim]"
        )
        self.console.print(
            "[dim]This may take 30–120 seconds depending on model size.[/dim]\n"
        )

        # Run with live spinner
        result: Optional[BenchmarkResult] = None
        with self.console.status(
            f"[cyan]Benchmarking {model.name}...[/cyan]", spinner="dots"
        ):
            result = run_benchmark(model.name, preset, warmup=True)

        if result.error:
            self.console.print(f"\n[red]Benchmark failed:[/red] {result.error}")
            return

        # Store result for comparison table
        key = f"{model.name}:{preset.value}"
        self.benchmark_results[key] = result

        # Display result
        score_color = {
            "A+": "bright_green",
            "A": "green",
            "B": "yellow",
            "C": "orange3",
            "F": "red",
        }.get(result.score_label, "white")

        res_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        res_table.add_column("Metric", style="bold cyan", min_width=24)
        res_table.add_column("Value", style="white")

        res_table.add_row("Model", f"[bold]{result.model_name}[/bold]")
        res_table.add_row("Preset", result.preset.value.capitalize())
        res_table.add_row(
            "Tokens/sec", f"[bold]{result.tokens_per_sec:.1f}[/bold] tok/s"
        )
        res_table.add_row(
            "First Token Latency", f"{result.first_token_latency_ms:.0f} ms"
        )
        res_table.add_row("Total Latency", f"{result.total_latency_ms:.0f} ms")
        res_table.add_row("Tokens Generated", str(result.total_tokens))
        res_table.add_row(
            "Performance Score",
            Text(f"  {result.score_label}  ", style=f"bold {score_color} on default"),
        )

        self.console.print(
            Panel(
                res_table,
                title="[bold]Benchmark Results[/bold]",
                border_style=score_color,
            )
        )

        # Offer prompt cost estimate
        self.console.print()
        tokens_str = Prompt.ask(
            "[cyan]Estimate cost for how many tokens?[/cyan]", default="1000"
        )
        try:
            estimate_tokens = int(tokens_str.replace(",", ""))
            cost = estimate_prompt_cost(estimate_tokens, result.tokens_per_sec)
            self.console.print(
                f"  [dim]→[/dim] {estimate_tokens:,} tokens at {cost['tokens_per_sec']} tok/s "
                f"≈ [bold]{cost['seconds']:.1f}s[/bold] ({cost['minutes']:.2f} min)"
            )
        except ValueError:
            pass

    def _render_comparison_table(self) -> None:
        """Show a side-by-side comparison of all benchmark results so far."""
        if not self.benchmark_results:
            self.console.print(
                "[yellow]No benchmark results yet. Run benchmarks first.[/yellow]"
            )
            return

        self.console.print()
        self.console.print(Rule("[bold cyan]📊 Benchmark Comparison Table[/bold cyan]"))
        self.console.print()

        table = Table(
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("Model", min_width=22, no_wrap=True)
        table.add_column("Preset", min_width=10)
        table.add_column("VRAM Est.", justify="right", min_width=10)
        table.add_column("Tok/s", justify="right", min_width=8)
        table.add_column("TTFT", justify="right", min_width=9)
        table.add_column("Latency", justify="right", min_width=10)
        table.add_column("Score", justify="center", min_width=7)

        vram_avail = (
            self.system_info.gpu.vram_total_gb
            if (self.system_info and self.system_info.gpu.is_available)
            else 0.0
        )

        score_styles = {
            "A+": "bright_green",
            "A": "green",
            "B": "yellow",
            "C": "orange3",
            "F": "red",
        }

        for key, result in sorted(self.benchmark_results.items()):
            # Find the model for VRAM estimate
            model = next((m for m in self.models if m.name == result.model_name), None)
            if model:
                est = estimate_vram(model, 4096, vram_avail)
                vram_str = f"{est.total_vram_gb:.1f} GB"
            else:
                vram_str = "?"

            score_style = score_styles.get(result.score_label, "white")
            table.add_row(
                result.model_name,
                result.preset.value,
                vram_str,
                f"{result.tokens_per_sec:.1f}",
                f"{result.first_token_latency_ms:.0f}ms",
                f"{result.total_latency_ms:.0f}ms",
                Text(f" {result.score_label} ", style=f"bold {score_style}"),
            )

        self.console.print(table)

    # -----------------------------------------------------------------------
    # View: Recommender
    # -----------------------------------------------------------------------

    def _render_recommender(self) -> None:
        """Show scenario-based model recommendations."""
        if not self.models:
            self.console.print("[yellow]No models available.[/yellow]")
            return

        self.console.print()
        self.console.print(Rule("[bold cyan]🔥 Auto Model Recommender[/bold cyan]"))
        self.console.print()

        scenarios = [
            "chatbot",
            "rag",
            "coding",
            "reasoning",
            "multilingual",
            "vision",
            "embedding",
        ]
        for i, s in enumerate(scenarios, 1):
            self.console.print(f"  [cyan]{i}.[/cyan] {s.capitalize()}")
        self.console.print()

        try:
            choice = IntPrompt.ask("[cyan]Select scenario[/cyan]", default=1)
            choice = max(1, min(choice, len(scenarios)))
        except (ValueError, KeyboardInterrupt):
            return

        scenario = scenarios[choice - 1]
        recommendations = recommend_models(
            self.models, self.system_info, scenario, top_n=5
        )

        if not recommendations:
            self.console.print(
                f"[yellow]No models suitable for '{scenario}' found.[/yellow]"
            )
            return

        self.console.print(
            f"\n[bold]Top models for [magenta]{scenario}[/magenta] on your system:[/bold]\n"
        )

        for rank, (model, score, reason) in enumerate(recommendations, 1):
            medal = ["🥇", "🥈", "🥉", "4.", "5."][rank - 1]
            self.console.print(
                f"  {medal} [bold]{model.name}[/bold]  "
                f"[dim](score: {score:.0f}/100)[/dim]"
            )
            self.console.print(f"     [dim]{reason}[/dim]")
            caps = ", ".join(c.value for c in model.capabilities)
            self.console.print(f"     Tags: [italic]{caps}[/italic]")
            self.console.print()

    # -----------------------------------------------------------------------
    # View: RAG Pipeline
    # -----------------------------------------------------------------------

    def _render_rag_pipeline(self) -> None:
        """Interactive RAG pipeline estimator."""
        if not self.models:
            self.console.print("[yellow]No models available.[/yellow]")
            return

        self.console.print()
        self.console.print(Rule("[bold cyan]🔄 RAG Pipeline Estimator[/bold cyan]"))
        self.console.print()

        # Find embedding models
        from models import ModelCapability

        embedding_models = [
            m for m in self.models if ModelCapability.EMBEDDING in m.capabilities
        ]
        other_models = [
            m for m in self.models if ModelCapability.EMBEDDING not in m.capabilities
        ]

        self.console.print("[bold]Available Embedding Models:[/bold]")
        if embedding_models:
            for i, m in enumerate(embedding_models, 1):
                self.console.print(f"  [cyan]{i}.[/cyan] {m.name}")
        else:
            self.console.print(
                "  [dim]None found — will estimate without embedding model[/dim]"
            )

        self.console.print("\n[bold]Available Generation Models:[/bold]")
        for i, m in enumerate(other_models[:10], 1):
            self.console.print(f"  [cyan]{i}.[/cyan] {m.name}")
        self.console.print()

        emb_model = None
        if embedding_models:
            try:
                ec = IntPrompt.ask(
                    "[cyan]Select embedding model (0 to skip)[/cyan]", default=0
                )
                if 1 <= ec <= len(embedding_models):
                    emb_model = embedding_models[ec - 1]
            except (ValueError, KeyboardInterrupt):
                pass

        gen_model = None
        if other_models:
            try:
                gc = IntPrompt.ask("[cyan]Select generation model[/cyan]", default=1)
                gc = max(1, min(gc, len(other_models[:10])))
                gen_model = other_models[gc - 1]
            except (ValueError, KeyboardInterrupt):
                pass

        # Find benchmark result for generation model if available
        bench_result = None
        if gen_model:
            key = f"{gen_model.name}:{BenchmarkPreset.SHORT.value}"
            bench_result = self.benchmark_results.get(key)

        estimate = estimate_rag_pipeline(
            emb_model, gen_model, self.system_info, bench_result
        )

        rag_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        rag_table.add_column("Component", style="bold cyan", min_width=26)
        rag_table.add_column("Value", style="white")

        if estimate.embedding_model:
            rag_table.add_row("Embedding Model", estimate.embedding_model)
            rag_table.add_row("Embedding VRAM", f"{estimate.embedding_vram_gb:.2f} GB")
        if estimate.generation_model:
            rag_table.add_row("Generation Model", estimate.generation_model)
            rag_table.add_row(
                "Generation VRAM", f"{estimate.generation_vram_gb:.2f} GB"
            )

        total_vram = estimate.embedding_vram_gb + estimate.generation_vram_gb
        rag_table.add_row("Total Pipeline VRAM", f"[bold]{total_vram:.2f} GB[/bold]")
        rag_table.add_row("", "")
        rag_table.add_row(
            "Retrieval Latency", f"{estimate.retrieval_latency_ms:.0f} ms"
        )
        rag_table.add_row("Rerank Latency", f"{estimate.rerank_latency_ms:.0f} ms")
        rag_table.add_row(
            "Generation Latency", f"{estimate.generation_latency_ms:.0f} ms"
        )
        rag_table.add_row(
            "Total Pipeline Latency",
            f"[bold]{estimate.total_pipeline_latency_ms:.0f} ms[/bold]",
        )

        self.console.print(
            Panel(
                rag_table,
                title="[bold]RAG Pipeline Estimate[/bold]",
                border_style="cyan",
            )
        )

    # -----------------------------------------------------------------------
    # View: Quantization Analysis
    # -----------------------------------------------------------------------

    def _render_quant_analysis(self) -> None:
        """Show quantization impact analysis across model variants."""
        self.console.print()
        self.console.print(
            Rule("[bold cyan]🧪 Quantization Impact Analysis[/bold cyan]")
        )
        self.console.print()

        groups = analyze_quantization_impact(self.models, self.system_info)

        if not groups:
            self.console.print(
                "[yellow]No model families with multiple quantization variants found.[/yellow]\n"
                "[dim]Install multiple variants, e.g.: ollama pull llama3:8b-q4_k_m && ollama pull llama3:8b-q8_0[/dim]"
            )
            return

        for group in groups:
            table = Table(
                box=box.ROUNDED,
                border_style="cyan",
                header_style="bold cyan",
                title=f"[bold]{group['base']}[/bold]",
                padding=(0, 1),
            )
            table.add_column("Model", min_width=30, no_wrap=True)
            table.add_column("Quant", min_width=10)
            table.add_column("VRAM", justify="right", min_width=9)
            table.add_column("Quality", justify="right", min_width=9)
            table.add_column("Speed", justify="right", min_width=8)
            table.add_column("Fit", justify="center", min_width=5)

            for v in group["variants"]:
                from models import GPUFitStatus

                fit_enum = GPUFitStatus(v["fit_status"])
                icon, color = FIT_ICONS[fit_enum]
                q_color = _quant_color(QuantizationType(v["quantization"].lower()))
                table.add_row(
                    v["name"],
                    Text(v["quantization"].upper(), style=q_color),
                    f"{v['vram_gb']:.1f} GB",
                    f"{v['quality_score']}/100",
                    f"{v['speed_score']}/100",
                    Text(icon, style=color),
                )

            self.console.print(table)
            self.console.print()

    # -----------------------------------------------------------------------
    # Main Menu
    # -----------------------------------------------------------------------

    def _print_main_menu(self) -> None:
        """Render the main navigation menu."""
        self.console.print()
        self.console.print(
            Panel(
                Align.center(
                    Text(f"{APP_TITLE}  v{APP_VERSION}", style="bold magenta")
                ),
                border_style="magenta",
                padding=(0, 2),
            )
        )
        self.console.print()

        menu_items = [
            ("1", "📦", "Model List", "All installed models with metadata"),
            ("2", "⚙️", "System Info", "GPU, RAM, CPU details"),
            ("3", "🧮", "VRAM Calculator", "VRAM estimation & KV cache scaling"),
            ("4", "⚡", "Benchmark", "Run performance benchmarks"),
            ("5", "📊", "Comparison Table", "Side-by-side benchmark results"),
            ("6", "🔥", "Recommender", "Best models for your use-case"),
            ("7", "🔄", "RAG Pipeline", "Estimate RAG pipeline resources"),
            ("8", "🧪", "Quant Analysis", "Compare quantization variants"),
            ("r", "🔄", "Refresh", "Reload models and system info"),
            ("q", "🚪", "Quit", "Exit the application"),
        ]

        table = Table(
            box=box.SIMPLE,
            show_header=False,
            padding=(0, 2),
            border_style="dim",
        )
        table.add_column("Key", style="bold cyan", width=4)
        table.add_column("Icon", width=3)
        table.add_column("Action", style="bold", min_width=18)
        table.add_column("Description", style="dim")

        for key, icon, action, desc in menu_items:
            table.add_row(f"[{key}]", icon, action, desc)

        self.console.print(Padding(table, (0, 4)))

        # Status bar
        model_count = len(self.models)
        gpu_str = (
            f"GPU: {self.system_info.gpu.name} ({self.system_info.gpu.vram_total_gb:.0f}GB)"
            if self.system_info and self.system_info.gpu.is_available
            else "GPU: CPU only"
        )
        ollama_str = "Ollama: ●" if self._ollama_running else "Ollama: ○"
        ollama_style = "green" if self._ollama_running else "red"

        status = Text()
        status.append(f" Models: {model_count} ", style="dim")
        status.append(" │ ", style="dim")
        status.append(gpu_str, style="dim")
        status.append(" │ ", style="dim")
        status.append(ollama_str, style=ollama_style)
        self.console.print(Align.center(status))
        self.console.print()

    def run(self) -> None:
        """Main application loop — loads data and handles user navigation."""
        self.console.clear()
        self.console.print(
            Panel(
                Align.center(Text(f"Loading {APP_TITLE}...", style="bold magenta")),
                border_style="magenta",
            )
        )
        self._load_data()

        while True:
            self.console.clear()
            self._print_main_menu()

            try:
                choice = (
                    Prompt.ask(
                        "[bold cyan]Enter choice[/bold cyan]",
                        choices=["1", "2", "3", "4", "5", "6", "7", "8", "r", "q"],
                        show_choices=False,
                    )
                    .lower()
                    .strip()
                )
            except (KeyboardInterrupt, EOFError):
                break

            if choice == "q":
                break
            elif choice == "r":
                self.console.print("[dim]Refreshing...[/dim]")
                self._load_data()
            elif choice == "1":
                self._render_model_list()
                self._pause()
            elif choice == "2":
                self._render_system_info()
                self._pause()
            elif choice == "3":
                self._render_vram_calculator()
                self._pause()
            elif choice == "4":
                self._render_benchmark()
                self._pause()
            elif choice == "5":
                self._render_comparison_table()
                self._pause()
            elif choice == "6":
                self._render_recommender()
                self._pause()
            elif choice == "7":
                self._render_rag_pipeline()
                self._pause()
            elif choice == "8":
                self._render_quant_analysis()
                self._pause()

        self.console.print("\n[bold magenta]Goodbye! 👋[/bold magenta]\n")

    def _pause(self) -> None:
        """Wait for user to press Enter before returning to main menu."""
        self.console.print()
        try:
            Prompt.ask("[dim]Press Enter to return to menu[/dim]", default="")
        except (KeyboardInterrupt, EOFError):
            pass
