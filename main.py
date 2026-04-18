"""
Ollama TUI - A Terminal User Interface for managing and analyzing Ollama models.

This is the main entry point that initializes the application and sets up
the Rich-based TUI with all panels and interactive features.
"""

import signal
import sys

from rich.console import Console
from rich.prompt import Prompt

from app import OllamaTUIApp
from utils.logger import setup_logger

logger = setup_logger(__name__)
console = Console()


def handle_interrupt(signum: int, frame) -> None:
    """Handle SIGINT (Ctrl+C) gracefully to avoid ugly tracebacks."""
    console.print("\n[yellow]Exiting Ollama TUI... Goodbye! 👋[/yellow]")
    sys.exit(0)


def main() -> None:
    """Main entry point for the Ollama TUI application.

    Initializes signal handlers, creates the app instance,
    and starts the interactive TUI loop.
    """
    signal.signal(signal.SIGINT, handle_interrupt)

    try:
        app = OllamaTUIApp()
        app.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting Ollama TUI... Goodbye! 👋[/yellow]")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        console.print("[dim]Check logs for details.[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()
