#!/usr/bin/env python3
"""
auto_deduplicate_documents.py — Automatically deduplicate PDF and Word documents
Detects duplicates by file hash, keeps one original, backs up and deletes duplicates.
No user review needed.

Usage:
  python auto_deduplicate_documents.py <source_folder> [--target /path/to/backup_folder]
  python auto_deduplicate_documents.py C:\MyDocuments\Photos --target C:\MyDocuments\Backup
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

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Document file extensions to process
DOCUMENT_EXTENSIONS = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'}


def get_file_hash(file_path: Path, chunk_size: int = 65536) -> str:
    """Calculate SHA-256 hash of a file."""
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


class DocumentDeduplicator:
    """Automatically deduplicate documents using file hash."""

    def __init__(self, source_folder: str, target_folder: str = None):
        self.source_folder = Path(source_folder)
        
        if not self.source_folder.exists():
            raise ValueError(f"Source folder '{source_folder}' not found")

        # Default target: parent_folder/documents_backup_TIMESTAMP
        if target_folder:
            self.target_folder = Path(target_folder)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.target_folder = self.source_folder.parent / f"documents_backup_{timestamp}"

        self.stats = {
            "total_scanned": 0,
            "total_groups": 0,
            "kept_count": 0,
            "deleted_count": 0,
            "backed_up_files": [],
            "group_log": [],
            "errors": []
        }

    def scan_documents(self) -> dict:
        """Scan folder for documents and group by file hash."""
        console.print(f"\n[bold cyan]Scanning for documents in:{/] {self.source_folder}\n")
        
        # Find all document files recursively
        document_files = []
        for ext in DOCUMENT_EXTENSIONS:
            document_files.extend(self.source_folder.rglob(f'*{ext}'))
        
        self.stats["total_scanned"] = len(document_files)
        console.print(f"[dim]Found {len(document_files)} documents[/]\n")
        
        if len(document_files) == 0:
            return {}
        
        # Group by file hash
        hash_groups = defaultdict(list)
        
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Calculating hashes...", total=len(document_files))
            
            for doc_path in document_files:
                try:
                    file_hash = get_file_hash(doc_path)
                    hash_groups[file_hash].append(str(doc_path))
                    progress.update(task, description=f"Processing: {doc_path.name[:40]}")
                except Exception as e:
                    logger.error(f"Error hashing {doc_path}: {e}")
                    self.stats["errors"].append(f"Hash failed: {doc_path.name}")
                finally:
                    progress.advance(task)
        
        # Keep only groups with duplicates (2+ files)
        duplicate_groups = {h: files for h, files in hash_groups.items() if len(files) > 1}
        self.stats["total_groups"] = len(duplicate_groups)
        
        return duplicate_groups

    def deduplicate(self, dry_run: bool = False) -> dict:
        """
        Deduplicate documents by:
        1. Keeping the first file in original location
        2. Copying duplicates to target folder (backup)
        3. Deleting duplicates from original location
        """
        console.print(f"\n[bold cyan]Starting Document Deduplication[/]")
        console.print(f"[dim]Source folder:[/] {self.source_folder}")
        console.print(f"[dim]Backup folder:[/] {self.target_folder}")
        console.print(f"[dim]Mode:[/] {'DRY RUN' if dry_run else 'DELETE & BACKUP'}\n")

        duplicate_groups = self.scan_documents()

        if len(duplicate_groups) == 0:
            console.print("[yellow]No duplicate documents found![/]\n")
            return self.stats

        if not dry_run:
            self.target_folder.mkdir(parents=True, exist_ok=True)

        files_to_delete = []

        # Phase 1: Plan deletions
        console.print(f"[bold yellow]Phase 1: Analyzing {len(duplicate_groups)} duplicate groups...[/]\n")
        
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Analyzing groups...", total=len(duplicate_groups))
            
            for file_hash, files in duplicate_groups.items():
                # Keep the first file (oldest or shortest path)
                files_sorted = sorted(files)
                kept_file = files_sorted[0]
                
                group_log = {
                    "hash": file_hash[:16] + "...",
                    "kept": kept_file,
                    "deleted": []
                }

                # All others are duplicates
                for duplicate_path in files_sorted[1:]:
                    src = Path(duplicate_path)
                    
                    # Preserve relative folder structure
                    try:
                        rel_path = src.relative_to(self.source_folder)
                    except ValueError:
                        rel_path = src.name
                    
                    backup_dest = self.target_folder / rel_path
                    files_to_delete.append((src, backup_dest, src.name, group_log))
                    group_log["deleted"].append(duplicate_path)
                
                self.stats["group_log"].append(group_log)
                progress.advance(task)

        # Phase 2: Summary
        console.print(f"\n[bold cyan]Phase 2: Summary[/]\n")
        
        summary_table = Table(show_header=True, header_style="bold cyan")
        summary_table.add_column("Metric", style="dim")
        summary_table.add_column("Count", style="yellow")
        summary_table.add_row("Total groups checked", str(len(duplicate_groups)))
        summary_table.add_row("Documents to keep", str(len(duplicate_groups)))
        summary_table.add_row("Duplicates to delete", str(len(files_to_delete)))
        summary_table.add_row("Errors found", str(len(self.stats["errors"])))
        
        console.print(summary_table)
        console.print()

        if len(files_to_delete) == 0:
            console.print("[yellow]No duplicates to delete![/]\n")
            return self.stats

        # Phase 3: Backup and delete
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
                        # Create backup directory
                        backup_dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(src), str(backup_dest))
                        
                        # Delete original
                        src.unlink()
                        
                        self.stats["backed_up_files"].append(str(backup_dest))
                        self.stats["deleted_count"] += 1
                        progress.update(task, description=f"Processing: {filename[:40]}")
                    except Exception as e:
                        logger.error(f"Failed to process {src}: {e}")
                        self.stats["errors"].append(f"Process failed: {src.name} - {str(e)}")
                    finally:
                        progress.advance(task)
        else:
            console.print(f"[bold yellow]DRY RUN: Would delete {len(files_to_delete)} files[/]\n")
            self.stats["deleted_count"] = len(files_to_delete)

        self.stats["kept_count"] = len(duplicate_groups)

        # Final report
        console.print(f"\n[bold]{'='*60}[/]")
        console.print(f"[bold]DOCUMENT DEDUPLICATION REPORT[/]")
        console.print(f"[bold]{'='*60}[/]\n")
        
        result_table = Table(show_header=True, header_style="bold cyan")
        result_table.add_column("Metric", style="dim")
        result_table.add_column("Count", style="yellow")
        result_table.add_row("Total documents scanned", str(self.stats["total_scanned"]))
        result_table.add_row("Duplicate groups found", str(self.stats["total_groups"]))
        result_table.add_row("Documents kept (original)", str(self.stats["kept_count"]))
        result_table.add_row("Duplicates deleted", str(self.stats["deleted_count"]))
        result_table.add_row("Backed up to folder", str(self.stats["deleted_count"]))
        if self.stats["errors"]:
            result_table.add_row("Errors found", str(len(self.stats["errors"])))
        
        console.print(result_table)
        console.print()

        if self.stats["errors"]:
            console.print("[bold red]Errors:[/]")
            for err in self.stats["errors"]:
                console.print(f"  [red]✗[/] {err}")
            console.print()

        console.print(f"[dim]Backup folder:[/] {self.target_folder}\n")

        # Save report
        report_path = self.target_folder / "document_deduplication_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        console.print(f"[green]Report saved:[/] {report_path}\n")

        return self.stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Automatically deduplicate PDF and Word documents")
    parser.add_argument("source_folder", help="Source folder to scan")
    parser.add_argument("--target", help="Target backup folder (default: source_parent/documents_backup_TIMESTAMP)")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be deleted without making changes")
    
    args = parser.parse_args()
    
    try:
        dedup = DocumentDeduplicator(args.source_folder, args.target)
        stats = dedup.deduplicate(dry_run=args.dry_run)
        
        sys.exit(0 if not stats["errors"] else 1)
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
