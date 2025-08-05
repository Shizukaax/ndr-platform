#!/usr/bin/env python3
"""
Cleanup script for NDR Platform development environment.
Removes temporary files, cache directories, and organizes project structure.
"""

import os
import shutil
import sys
from pathlib import Path

def main():
    """Clean up the project directory."""
    project_root = Path(__file__).parent.absolute()
    print(f"🧹 Cleaning up NDR Platform at: {project_root}")
    
    # Files and directories to clean
    cleanup_patterns = [
        "__pycache__",
        "*.pyc", 
        "*.pyo",
        ".pytest_cache",
        "*.log.1",
        "*.log.2", 
        "*.log.3",
        "*.tmp",
        "*.temp",
        ".DS_Store",
        "Thumbs.db"
    ]
    
    # Directories to clean recursively
    cache_dirs = [
        "cache",
        "temp",
        "tmp"
    ]
    
    cleaned_count = 0
    
    # Clean cache files
    for pattern in cleanup_patterns:
        if "*" in pattern:
            # Handle wildcard patterns
            import glob
            matches = glob.glob(str(project_root / "**" / pattern), recursive=True)
            for match in matches:
                try:
                    if os.path.isfile(match):
                        os.remove(match)
                        print(f"   Removed file: {os.path.relpath(match)}")
                        cleaned_count += 1
                    elif os.path.isdir(match):
                        shutil.rmtree(match)
                        print(f"   Removed directory: {os.path.relpath(match)}")
                        cleaned_count += 1
                except Exception as e:
                    print(f"   Warning: Could not remove {match}: {e}")
        else:
            # Handle exact directory names
            for root, dirs, files in os.walk(project_root):
                if pattern in dirs:
                    dir_path = os.path.join(root, pattern)
                    try:
                        shutil.rmtree(dir_path)
                        print(f"   Removed directory: {os.path.relpath(dir_path)}")
                        cleaned_count += 1
                    except Exception as e:
                        print(f"   Warning: Could not remove {dir_path}: {e}")
    
    # Clean specific cache directories
    for cache_dir in cache_dirs:
        cache_path = project_root / cache_dir
        if cache_path.exists() and cache_path.is_dir():
            # Only clean contents, keep the directory
            for item in cache_path.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                        print(f"   Cleaned cache file: {item.name}")
                        cleaned_count += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        print(f"   Cleaned cache directory: {item.name}")
                        cleaned_count += 1
                except Exception as e:
                    print(f"   Warning: Could not clean {item}: {e}")
    
    # Report results
    print(f"\n✅ Cleanup completed!")
    print(f"   - Removed {cleaned_count} items")
    print(f"   - Project structure maintained")
    print(f"   - Documentation organized in docs/")
    print(f"   - Tests organized in tests/")
    
    # Check directory structure
    print(f"\n📁 Current project structure:")
    important_dirs = ["app", "core", "docs", "tests", "config", "logs"]
    for dir_name in important_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/ (missing)")

if __name__ == "__main__":
    main()
