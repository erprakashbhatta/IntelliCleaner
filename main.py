#!/usr/bin/env python3
"""
main.py — PhotoMind Entry Point
Run the web UI or use the CLI for headless operation.

Usage:
  python main.py                         # Launch web UI (default)
  python main.py --folder /path/photos   # Headless CLI scan
  python main.py --help
"""
import argparse
import logging
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# Suppress noisy third-party loggers
for noisy in ["httpx", "httpcore", "openai", "urllib3", "PIL"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


def run_ui():
    """Launch the Flask + SocketIO web server and open the browser."""
    from server import run_server
    from config import UI_HOST, UI_PORT
    import threading, webbrowser, time

    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://{UI_HOST}:{UI_PORT}")

    threading.Thread(target=open_browser, daemon=True).start()
    run_server()


def run_cli(args):
    """Headless pipeline: scan → group → analyze → auto-decide or print report."""
    from rich.console import Console
    from rich.table   import Table
    from pipeline     import Pipeline

    console = Console()
    console.print(f"\n[bold]PhotoMind CLI[/]  |  folder: [cyan]{args.folder}[/]\n")

    pipe   = Pipeline(args.folder)
    result = pipe.run()

    pipe.pipeline_result_analyses_ref = result.analyses

    console.print(f"\n[bold]Results:[/]")
    console.print(f"  Images scanned : [cyan]{result.total_images}[/]")
    console.print(f"  Duplicate groups: [yellow]{result.total_groups}[/]")
    console.print(f"  Unique images  : [green]{len(result.singletons)}[/]")
    console.print(f"  Scan time      : {result.scan_time:.1f}s\n")

    if args.report:
        table = Table(title="Duplicate Groups", show_lines=True)
        table.add_column("ID",     style="dim",    width=14)
        table.add_column("Tier",   style="yellow", width=10)
        table.add_column("Count",  justify="right")
        table.add_column("Sim%",   justify="right")
        table.add_column("AI Pick",             width=30)
        table.add_column("Confidence", justify="right")

        for grp in result.groups:
            analysis = result.analyses.get(grp.group_id)
            best_name = ""
            conf      = ""
            if analysis:
                idx = analysis.best_idx
                if 0 <= idx < len(grp.images):
                    best_name = grp.images[idx].filename
                conf = f"{analysis.confidence:.0%}"
            table.add_row(
                grp.group_id[-8:],
                grp.tier,
                str(len(grp.images)),
                f"{grp.similarity:.0%}",
                best_name,
                conf,
            )
        console.print(table)

    if args.auto_decide and args.output:
        console.print(f"\n[bold]Auto-applying high-confidence decisions…[/]")
        from config import LLM_CONFIDENCE_AUTO_THRESHOLD
        applied = 0
        for grp in result.groups:
            analysis = result.analyses.get(grp.group_id)
            if not analysis or analysis.confidence < LLM_CONFIDENCE_AUTO_THRESHOLD:
                continue
            summary = pipe.apply_decision(
                group=grp,
                kept_idx=analysis.best_idx,
                reason_tags=["auto_decision"],
                human_notes=f"Auto CLI (conf={analysis.confidence:.2f})",
                output_folder=args.output,
                dry_run=args.dry_run,
            )
            applied += 1
        verb = "Would apply" if args.dry_run else "Applied"
        console.print(f"  {verb} [green]{applied}[/] decisions")

    if not args.auto_decide:
        console.print(
            "\n[dim]Tip: run [bold]python main.py[/] (no args) to open the web UI for manual review.[/]"
        )


def main():
    parser = argparse.ArgumentParser(
        description="PhotoMind — AI-powered photo deduplication with continuous learning"
    )
    parser.add_argument("--folder",      type=str, help="Folder to scan (CLI mode)")
    parser.add_argument("--output",      type=str, help="Output folder for deduplicated photos")
    parser.add_argument("--report",      action="store_true", help="Print group report table")
    parser.add_argument("--auto-decide", action="store_true", help="Auto-apply high-confidence decisions")
    parser.add_argument("--dry-run",     action="store_true", help="Preview only, don't move files")
    parser.add_argument("--ui",          action="store_true", help="Force launch web UI")

    args = parser.parse_args()

    if args.ui or not args.folder:
        run_ui()
    else:
        if not Path(args.folder).is_dir():
            print(f"Error: '{args.folder}' is not a valid directory", file=sys.stderr)
            sys.exit(1)
        run_cli(args)


if __name__ == "__main__":
    main()
