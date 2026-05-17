"""
apply_decisions.py — Apply duplicate removal decisions and organize files
Copies kept photos to target folder with same structure and generates a report
"""
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────
from config import SCANS_DIR

class DecisionApplier:
    def __init__(self, scan_id: str, target_folder: str = None):
        self.scan_id = scan_id
        self.scan_dir = SCANS_DIR / scan_id
        
        if not self.scan_dir.exists():
            raise ValueError(f"Scan '{scan_id}' not found in {SCANS_DIR}")
        
        # Load scan metadata
        with open(self.scan_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        
        self.source_folder = self.metadata["folder"]
        self.target_folder = Path(target_folder) if target_folder else Path(self.source_folder).parent / f"{Path(self.source_folder).name}_cleaned"
        
        # Statistics
        self.stats = {
            "total_scanned": self.metadata["total_images"],
            "total_groups": self.metadata["total_groups"],
            "deleted_count": 0,
            "transferred_count": 0,
            "deleted_files": [],
            "transferred_files": [],
            "errors": []
        }
    
    def load_decisions(self):
        """Load all group decisions from the analysis results."""
        analyses_file = self.scan_dir / "analyses.json"
        if not analyses_file.exists():
            logger.warning("No analyses file found - no decisions have been made yet")
            return {}
        
        with open(analyses_file) as f:
            return json.load(f)
    
    def load_records(self):
        """Load all image records."""
        with open(self.scan_dir / "records.json") as f:
            return json.load(f)
    
    def load_groups(self):
        """Load all duplicate groups."""
        with open(self.scan_dir / "groups.json") as f:
            return json.load(f)
    
    @staticmethod
    def _auto_best_idx(group: dict, quality_map: dict, size_map: dict) -> int:
        """Pick best image index: highest quality score → largest size → index 0."""
        images = group["images"]
        return max(
            range(len(images)),
            key=lambda i: (quality_map.get(images[i], 0.0), size_map.get(images[i], 0)),
        )

    def apply_decisions(self, dry_run=False):
        """
        Apply all decisions:
        - For groups with an LLM/human decision → use that best_idx
        - For groups without any decision → auto-select by quality score then file size
        - Copy kept files to target folder (preserving folder structure)
        - Delete duplicate files from source
        """
        logger.info(f"{'DRY RUN: ' if dry_run else ''}Applying decisions from scan '{self.scan_id}'")
        logger.info(f"Source folder: {self.source_folder}")
        logger.info(f"Target folder: {self.target_folder}")

        # Create target folder
        if not dry_run:
            self.target_folder.mkdir(parents=True, exist_ok=True)

        # Load all data
        decisions = self.load_decisions()   # may be empty dict
        records   = self.load_records()
        groups    = self.load_groups()

        # Quick-lookup maps for auto-selection
        quality_map = {r["path"]: float(r.get("quality_score") or 0.0) for r in records}
        size_map    = {r["path"]: int(r.get("size_bytes")    or 0)     for r in records}

        # Build a map: file path -> group
        groups_by_path: dict[str, dict] = {}
        for group in groups:
            for path in group["images"]:
                groups_by_path[path] = group

        files_to_keep:   set[str] = set()
        files_to_delete: set[str] = set()
        auto_count = 0
        manual_count = 0

        for group in groups:
            images   = group["images"]
            decision = decisions.get(group["group_id"])

            if decision and decision.get("best_idx") is not None:
                best_idx = decision["best_idx"]
                if not (0 <= best_idx < len(images)):
                    best_idx = self._auto_best_idx(group, quality_map, size_map)
                    auto_count += 1
                else:
                    manual_count += 1
            else:
                best_idx = self._auto_best_idx(group, quality_map, size_map)
                auto_count += 1

            files_to_keep.add(images[best_idx])
            for i, path in enumerate(images):
                if i != best_idx:
                    files_to_delete.add(path)

        logger.info(f"Groups: {manual_count} manual decisions, {auto_count} auto-selected")

        # Also include singletons (files not in any group) - keep them as-is
        for record in records:
            path = record["path"]
            if path not in groups_by_path:
                files_to_keep.add(path)
        
        # Copy kept files to target folder
        logger.info(f"\nCopying {len(files_to_keep)} files to target folder...")
        for file_path in sorted(files_to_keep):
            try:
                src = Path(file_path)
                if not src.exists():
                    logger.warning(f"File not found: {file_path}")
                    self.stats["errors"].append(f"Source file not found: {file_path}")
                    continue
                
                # Calculate relative path from source folder
                rel_path = src.relative_to(self.source_folder)
                dst = self.target_folder / rel_path
                
                # Create destination directory
                if not dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(dst))
                
                self.stats["transferred_count"] += 1
                self.stats["transferred_files"].append(str(rel_path))
                
                if not dry_run:
                    logger.info(f"✓ Copied: {rel_path}")
            
            except Exception as e:
                logger.error(f"Failed to copy {file_path}: {e}")
                self.stats["errors"].append(f"Copy failed: {file_path} - {str(e)}")
        
        # Collect deleted files info
        logger.info(f"\nMarking {len(files_to_delete)} files for deletion...")
        for file_path in sorted(files_to_delete):
            try:
                src = Path(file_path)
                if src.exists():
                    rel_path = src.relative_to(self.source_folder)
                    self.stats["deleted_count"] += 1
                    self.stats["deleted_files"].append(str(rel_path))
                    
                    if not dry_run:
                        src.unlink()  # Delete the file
                        logger.info(f"✓ Deleted: {rel_path}")
            
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
                self.stats["errors"].append(f"Delete failed: {file_path} - {str(e)}")
        
        return True
    
    def generate_report(self, output_file=None):
        """Generate a comprehensive report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "scan_id": self.scan_id,
            "source_folder": str(self.source_folder),
            "target_folder": str(self.target_folder),
            "stats": {
                "total_scanned": self.stats["total_scanned"],
                "total_groups": self.stats["total_groups"],
                "deleted_count": self.stats["deleted_count"],
                "transferred_count": self.stats["transferred_count"],
                "remaining_count": self.stats["total_scanned"] - self.stats["deleted_count"],
                "error_count": len(self.stats["errors"])
            },
            "deleted_files": self.stats["deleted_files"],
            "transferred_files": self.stats["transferred_files"],
            "errors": self.stats["errors"]
        }
        
        # Save report
        if output_file is None:
            output_file = self.scan_dir / f"decision_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            output_file = Path(output_file)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Report saved to: {output_file}")
        
        return report
    
    def print_summary(self):
        """Print a human-readable summary."""
        stats = self.stats
        
        print("\n" + "="*70)
        print("DUPLICATE REMOVAL SUMMARY".center(70))
        print("="*70)
        print(f"\nSource Folder: {self.source_folder}")
        print(f"Target Folder: {self.target_folder}")
        print(f"\n{'Statistic':<40} {'Value':>15}")
        print("-"*70)
        print(f"{'Total Images Scanned':<40} {stats['total_scanned']:>15}")
        print(f"{'Duplicate Groups Found':<40} {stats['total_groups']:>15}")
        print(f"{'Files Deleted (Duplicates)':<40} {stats['deleted_count']:>15}")
        print(f"{'Files Transferred to Target':<40} {stats['transferred_count']:>15}")
        print(f"{'Files Remaining':<40} {stats['total_scanned'] - stats['deleted_count']:>15}")
        print(f"{'Errors':<40} {len(stats['errors']):>15}")
        print("-"*70)
        
        # Show deleted files
        if stats['deleted_files']:
            print(f"\n📋 DELETED FILES ({len(stats['deleted_files'])} total):\n")
            for i, fname in enumerate(sorted(stats['deleted_files']), 1):
                print(f"  {i:3}. {fname}")
        
        # Show errors if any
        if stats['errors']:
            print(f"\n⚠️  ERRORS ({len(stats['errors'])} total):\n")
            for error in stats['errors']:
                print(f"  • {error}")
        
        print("\n" + "="*70)
        print(f"✓ Operation completed successfully!")
        print("="*70 + "\n")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python apply_decisions.py <scan_id> [target_folder] [--dry-run]")
        print("\nExample:")
        print("  python apply_decisions.py Downloads_20260506_170000")
        print("  python apply_decisions.py Downloads_20260506_170000 /path/to/cleaned --dry-run")
        sys.exit(1)
    
    scan_id = sys.argv[1]
    target_folder = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    dry_run = '--dry-run' in sys.argv
    
    try:
        applier = DecisionApplier(scan_id, target_folder)
        
        if dry_run:
            logger.info("Running in DRY-RUN mode - no files will be modified")
        
        # Apply decisions
        applier.apply_decisions(dry_run=dry_run)
        
        # Generate report
        report = applier.generate_report()
        
        # Print summary
        applier.print_summary()
        
        # Return report as JSON for programmatic use
        return report
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
