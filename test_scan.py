#!/usr/bin/env python3
"""
Test script to verify duplicate detection is working
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import Pipeline
from rich.console import Console

console = Console()

def test_duplicate_detection():
    """Test the full pipeline on a folder"""
    
    # Try to find a test folder with images
    test_folders = [
        Path.home() / "Pictures",
        Path.home() / "Downloads",
        Path("C:\\Users\\raul_\\Pictures"),
        Path("C:\\Users\\raul_\\Downloads"),
    ]
    
    folder = None
    for f in test_folders:
        if f.exists():
            folder = f
            break
    
    if not folder:
        console.print("[red]❌ No test folder found. Please provide a folder path.[/]")
        print("Usage: python test_scan.py /path/to/images")
        return False
    
    # Use command line argument if provided
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    
    if not folder.exists():
        console.print(f"[red]❌ Folder does not exist: {folder}[/]")
        return False
    
    console.print(f"[bold cyan]🔍 Testing duplicate detection on: {folder}[/]\n")
    
    # Run the pipeline
    pipeline = Pipeline(str(folder))
    result = pipeline.run()
    
    console.print("\n[bold]=== RESULTS ===[/]\n")
    console.print(f"Total images scanned:     {result.total_images}")
    console.print(f"Total duplicate groups:   {result.total_groups}")
    console.print(f"Unique images (no dups):  {len(result.singletons)}")
    console.print(f"Scan time:                {result.scan_time:.1f}s\n")
    
    if result.total_groups == 0:
        console.print("[yellow]⚠️  No duplicates found. This could be normal if all images are unique.[/]\n")
    else:
        console.print("[bold green]✓ Duplicates found![/]\n")
        
        # Show breakdown by tier
        tier_counts = {}
        for g in result.groups:
            tier_counts[g.tier] = tier_counts.get(g.tier, 0) + 1
        
        console.print("[bold]Breakdown by detection tier:[/]")
        for tier, count in sorted(tier_counts.items()):
            console.print(f"  • {tier:10s}: {count:3d} groups")
        
        console.print("\n[bold]Sample groups (first 3):[/]")
        for i, grp in enumerate(result.groups[:3]):
            console.print(f"\n  Group {i+1} ({grp.tier}, {len(grp.images)} images, {grp.similarity:.1%} similar)")
            for img in grp.images[:3]:
                console.print(f"    - {Path(img.path).name} ({img.width}×{img.height}, {img.size_bytes/1024/1024:.1f}MB)")
    
    # Test save functionality
    console.print("\n[bold cyan]💾 Testing save/load functionality...[/]\n")
    try:
        from server import _save_scan_results, _load_scan_results, _list_saved_scans
        
        # Save
        scan_id = _save_scan_results(str(folder), result)
        console.print(f"[green]✓ Saved scan: {scan_id}[/]")
        
        # List
        scans = _list_saved_scans()
        console.print(f"[green]✓ Total saved scans: {len(scans)}[/]")
        
        # Load
        loaded = _load_scan_results(scan_id)
        if loaded:
            folder_back, result_back = loaded
            console.print(f"[green]✓ Loaded scan successfully[/]")
            console.print(f"  - Folder: {folder_back}")
            console.print(f"  - Groups: {result_back.total_groups}")
            console.print(f"  - Images: {result_back.total_images}")
        else:
            console.print("[red]❌ Failed to load scan[/]")
    except Exception as e:
        console.print(f"[red]❌ Save/load test failed: {e}[/]")
        import traceback
        traceback.print_exc()
    
    return result.total_groups > 0

if __name__ == "__main__":
    success = test_duplicate_detection()
    sys.exit(0 if success else 1)
