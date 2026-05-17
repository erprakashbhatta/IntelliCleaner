#!/usr/bin/env python3
"""
auto_deduplicate.py — Fully automatic duplicate photo cleanup
Automatically selects the best image from each duplicate group and backs up all other duplicates.
Keeps one original, copies duplicates to the target folder, then removes duplicates from the source.
No thumbnails, no user review, no interactive decisions needed.

Usage:
  python auto_deduplicate.py <scan_id> [--target /path/to/backup_folder]
  python auto_deduplicate.py ViberPhotos_20260508_074902 --target ./duplicates_backup
"""
import json
import shutil
import logging
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

import sys as _sys
console = Console(highlight=False, emoji=False,
                  file=open(_sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1, closefd=False))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

from config import SCANS_DIR
from pipeline import Pipeline


class AutoDeduplicator:
    """Automatically deduplicate photos using LLM/quality score intelligence."""

    def __init__(self, scan_id: str, target_folder: str = None):
        self.scan_id = scan_id
        self.scan_dir = SCANS_DIR / scan_id

        if not self.scan_dir.exists():
            raise ValueError(f"Scan '{scan_id}' not found in {SCANS_DIR}")

        # Load metadata
        with open(self.scan_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        self.source_folder = self.metadata["folder"]
        
        # Default target: parent_folder/duplicates_backup_TIMESTAMP
        if target_folder:
            self.target_folder = Path(target_folder)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.target_folder = Path(self.source_folder).parent / f"duplicates_backup_{timestamp}"

        self.stats = {
            "total_scanned": self.metadata["total_images"],
            "total_groups": self.metadata["total_groups"],
            "kept_count": 0,
            "moved_count": 0,
            "moved_files": [],
            "kept_files": [],
            "group_log": [],
            "errors": []
        }

    def load_records(self) -> dict:
        """Load all image records if saved scan data exists."""
        records_file = self.scan_dir / "records.json"
        if not records_file.exists():
            return {}
        with open(records_file) as f:
            return {rec["path"]: rec for rec in json.load(f)}

    def load_groups(self) -> list:
        """Load all duplicate groups if saved scan data exists."""
        groups_file = self.scan_dir / "groups.json"
        if not groups_file.exists():
            return []
        with open(groups_file) as f:
            return json.load(f)

    def load_analyses(self) -> dict:
        """Load LLM analyses for groups if saved scan data exists."""
        analyses_file = self.scan_dir / "analyses.json"
        if not analyses_file.exists():
            return {}
        with open(analyses_file) as f:
            return json.load(f)

    def rebuild_scan_data(self) -> tuple[dict, list, dict]:
        """Run the pipeline on the source folder when saved scan data is missing."""
        logger.info("Saved scan data missing or incomplete — rebuilding from source folder...")
        console.print("[yellow]Saved scan data missing. Re-scanning source folder to rebuild groups…[/]")

        pipeline = Pipeline(self.source_folder)
        result   = pipeline.run()

        records: dict = {}
        all_recs = result.singletons + [img for group in result.groups for img in group.images]
        for rec in all_recs:
            records[rec.path] = {
                "path":          rec.path,
                "filename":      rec.filename,
                "size_bytes":    rec.size_bytes,
                "quality_score": float(rec.quality_score) if rec.quality_score is not None else 0.0,
            }

        groups: list = []
        for group in result.groups:
            groups.append({
                "group_id":      group.group_id,
                "tier":          group.tier,
                "similarity":    group.similarity,
                "images":        [img.path for img in group.images],
                "best_guess_idx": group.best_guess_idx,
                "decided":       group.decided,
            })

        analyses: dict = {}
        for group_id, analysis in result.analyses.items():
            analyses[group_id] = {
                "best_idx":        analysis.best_idx,
                "confidence":      analysis.confidence,
                "recommendation":  analysis.recommendation,
                "reasons":         analysis.reasons,
                "disqualifiers":   analysis.disqualifiers,
                "backend":         analysis.backend,
                "rag_context_used": analysis.rag_context_used,
            }

        # Persist so future runs skip the full re-scan
        self._save_rebuilt_data(records, groups, analyses, result)
        return records, groups, analyses

    def _save_rebuilt_data(self, records: dict, groups: list, analyses: dict, result):
        """Save rebuilt scan data to the scan directory."""
        import json as _json

        def _safe(v):
            import numpy as np
            if isinstance(v, np.generic):  return v.item()
            if isinstance(v, np.ndarray): return v.tolist()
            if isinstance(v, dict):  return {k: _safe(x) for k, x in v.items()}
            if isinstance(v, list):  return [_safe(x) for x in v]
            return v

        (self.scan_dir / "records.json").write_text(
            _json.dumps(_safe(list(records.values())), indent=2)
        )
        (self.scan_dir / "groups.json").write_text(
            _json.dumps(_safe(groups), indent=2)
        )
        (self.scan_dir / "analyses.json").write_text(
            _json.dumps(_safe(analyses), indent=2)
        )
        # Update metadata with refreshed counts
        try:
            meta = _json.loads((self.scan_dir / "metadata.json").read_text())
            meta["total_images"] = result.total_images
            meta["total_groups"] = result.total_groups
            meta["scan_time"]    = result.scan_time
            (self.scan_dir / "metadata.json").write_text(_json.dumps(meta, indent=2))
        except Exception:
            pass
        console.print(f"[green]Rebuilt scan data saved to {self.scan_dir}[/]")

    def select_best_image(self, group: dict, records: dict, analyses: dict) -> int:
        """
        Select the best image from a group using intelligent criteria:
        1. LLM recommendation (best_idx from analysis)
        2. Highest quality score
        3. First image (fallback)
        """
        group_id = group["group_id"]
        images = group["images"]

        # Try LLM recommendation first
        if group_id in analyses:
            analysis = analyses[group_id]
            best_idx = analysis.get("best_idx", 0)
            if 0 <= best_idx < len(images):
                logger.debug(f"  Group {group_id}: Using LLM recommendation (idx={best_idx})")
                return best_idx

        # Fallback: highest quality score, then largest file size
        best_idx = max(
            range(len(images)),
            key=lambda i: (
                records.get(images[i], {}).get("quality_score", 0.0),
                records.get(images[i], {}).get("size_bytes",    0),
            ),
        )
        logger.debug(f"  Group {group_id}: Auto-selected idx={best_idx}")
        return best_idx

    def deduplicate(self, dry_run: bool = False) -> dict:
        """
        Deduplicate by:
        1. Keeping the best image in original location
        2. Copying duplicates to target folder (backup)
        3. Deleting duplicates from original location
        """
        console.print(f"\n[bold cyan]Starting Auto-Deduplication (fully automatic)[/]")
        console.print(f"[dim]Source folder:[/] {self.source_folder}")
        console.print(f"[dim]Backup folder:[/] {self.target_folder}")
        console.print(f"[dim]Mode:[/] {'DRY RUN' if dry_run else 'DELETE & BACKUP'}\n")

        records  = self.load_records()
        groups   = self.load_groups()
        analyses = self.load_analyses()

        # Rebuild only when the data files are genuinely absent (empty list/dict is a valid result)
        records_missing = not (self.scan_dir / "records.json").exists()
        groups_missing  = not (self.scan_dir / "groups.json").exists()
        if records_missing or groups_missing:
            records, groups, analyses = self.rebuild_scan_data()

        if not dry_run:
            self.target_folder.mkdir(parents=True, exist_ok=True)

        # Track what we're keeping vs deleting
        files_to_keep = set()
        files_to_delete = []  # List of (source, backup_dest, filename) tuples

        # Phase 1: Analyze groups and plan deletions
        console.print(f"[bold yellow]Phase 1: Analyzing {len(groups)} duplicate groups...[/]\n")
        
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Analyzing groups...", total=len(groups))
            
            for group in groups:
                group_id = group["group_id"]
                images = group["images"]

                if len(images) < 2:
                    progress.advance(task)
                    continue  # Skip singletons

                # Select best image
                best_idx = self.select_best_image(group, records, analyses)
                best_path = Path(images[best_idx])

                files_to_keep.add(best_path)
                self.stats["kept_files"].append(str(best_path))

                group_log = {
                    "group_id": group_id,
                    "kept": str(best_path),
                    "deleted": []
                }

                # All others are duplicates to delete (and backup)
                for idx, img_path in enumerate(images):
                    if idx == best_idx:
                        continue

                    src = Path(img_path)
                    if not src.is_absolute():
                        # If path is relative, assume it's relative to source_folder
                        src = self.source_folder / src
                    
                    if not src.exists():
                        self.stats["errors"].append(f"Source not found: {src}")
                        continue

                    # Preserve relative folder structure in target (backup)
                    try:
                        rel_path = src.relative_to(self.source_folder)
                    except ValueError:
                        # If path is not under source_folder, use just the filename
                        rel_path = src.name
                    
                    backup_dest = self.target_folder / rel_path

                    files_to_delete.append((src, backup_dest, src.name, group_log))
                    group_log["deleted"].append(str(src))

                self.stats["group_log"].append(group_log)
                progress.advance(task)

        # Phase 2: Display summary before deleting
        console.print(f"\n[bold cyan]Phase 2: Summary[/]\n")
        
        summary_table = Table(show_header=True, header_style="bold cyan")
        summary_table.add_column("Metric", style="dim")
        summary_table.add_column("Count", style="yellow")
        summary_table.add_row("Total groups checked", str(len(groups)))
        summary_table.add_row("Images to keep (best)", str(len(files_to_keep)))
        summary_table.add_row("Duplicates to delete", str(len(files_to_delete)))
        summary_table.add_row("Errors found", str(len(self.stats["errors"])))
        
        console.print(summary_table)
        console.print()

        if len(files_to_delete) == 0:
            console.print("[yellow]No duplicates found to delete![/]\n")
            return self.stats

        console.print(f"[bold magenta]Selected keep file for first groups:[/]\n")
        for entry in self.stats["group_log"][:10]:
            console.print(f"  • Group {entry['group_id']}: keep {Path(entry['kept']).name}")
        if len(self.stats["group_log"]) > 10:
            console.print(f"  [dim]... plus {len(self.stats['group_log']) - 10} more groups[/]\n")

        # Phase 3: Backup and delete files
        if not dry_run:
            console.print(f"[bold cyan]Phase 3: Backing up and deleting files...[/]\n")
            
            with Progress(
                SpinnerColumn(),
                BarColumn(),
                TextColumn("[progress.description]{task.description}"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                task = progress.add_task("Processing files...", total=len(files_to_delete))
                
                for src, backup_dest, filename, group_log in files_to_delete:
                    try:
                        # First, backup to target folder
                        backup_dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(src), str(backup_dest))  # copy2 preserves metadata
                        
                        # Then delete from original location
                        src.unlink()
                        
                        self.stats["moved_files"].append(str(backup_dest))
                        self.stats["moved_count"] += 1
                        progress.update(task, description=f"Processing: {filename[:40]}")
                    except Exception as e:
                        logger.error(f"Failed to process {src}: {e}")
                        self.stats["errors"].append(f"Process failed: {src.name} → {str(e)}")
                    finally:
                        progress.advance(task)
        else:
            console.print(f"[bold yellow]DRY RUN MODE: Would delete {len(files_to_delete)} files[/]")
            console.print(f"[dim](Backed up to {self.target_folder})[/]\n")
            self.stats["moved_count"] = len(files_to_delete)

        self.stats["kept_count"] = len(files_to_keep)

        # Final Report
        console.print(f"\n[bold cyan]═══════════════════════════════════════════════════════════[/]")
        console.print(f"[bold cyan]DEDUPLICATION REPORT[/]")
        console.print(f"[bold cyan]═══════════════════════════════════════════════════════════[/]\n")

        result_table = Table(show_header=True, header_style="bold green")
        result_table.add_column("Metric", style="dim")
        result_table.add_column("Count", style="green")
        result_table.add_row("Total images scanned", str(self.stats["total_scanned"]))
        result_table.add_row("Duplicate groups found", str(self.stats["total_groups"]))
        result_table.add_row("Images kept (original)", str(self.stats["kept_count"]))
        result_table.add_row("Duplicates deleted",     str(self.stats["moved_count"]))
        result_table.add_row("Backed up to folder",    str(self.stats["moved_count"]))
        result_table.add_row("Groups logged",          str(len(self.stats["group_log"])))
        
        if self.stats["errors"]:
            result_table.add_row("[red]⚠ Errors[/]", f"[red]{len(self.stats['errors'])}[/]")
        
        console.print(result_table)
        console.print()
        
        console.print(f"[dim]Backup folder:[/] {self.target_folder}\n")

        if self.stats["errors"]:
            console.print(f"[bold red]Errors ({len(self.stats['errors'])}):[/]")
            for err in self.stats["errors"][:5]:  # Show first 5 errors
                console.print(f"  [red]✗[/] {err}")
            if len(self.stats["errors"]) > 5:
                console.print(f"  [dim]... and {len(self.stats['errors']) - 5} more[/]")
            console.print()

        return self.stats


    def save_report(self, output_file: str = None):
        """Save a detailed report of what was done."""
        if output_file is None:
            output_file = self.target_folder / "deduplication_report.json"
        else:
            output_file = Path(output_file)

        report = {
            "timestamp": datetime.now().isoformat(),
            "scan_id": self.scan_id,
            "source_folder": str(self.source_folder),
            "target_folder": str(self.target_folder),
            "stats": self.stats,
            "group_log": self.stats["group_log"],
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Automatically deduplicate photos - DELETE duplicates & BACKUP to target folder"
    )
    parser.add_argument("scan_id", help="Scan ID from SCANS_DIR")
    parser.add_argument(
        "--target",
        help="Target folder for backup (default: duplicates_backup_TIMESTAMP)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without deleting files",
    )
    parser.add_argument(
        "--report",
        help="Save report to file",
    )

    args = parser.parse_args()

    try:
        dedup = AutoDeduplicator(args.scan_id, args.target)
        dedup.deduplicate(dry_run=args.dry_run)

        if not args.dry_run:
            report_path = args.report if args.report else dedup.target_folder / "deduplication_report.json"
            dedup.save_report(report_path)
            console.print(f"[green]Report saved:[/] {report_path}\n")
        else:
            console.print("[yellow](DRY RUN — no files were actually deleted)[/]\n")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Deduplication cancelled by user[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()

