#!/usr/bin/env python3
"""
Fix encoding issues in Python files by replacing corrupted Unicode characters.
"""

import os
import re
from pathlib import Path

def fix_unicode_in_file(file_path):
    """Fix Unicode encoding issues in a single file"""
    try:
        # Read the file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Replace corrupted Unicode characters
        fixes = {
            'ð\x9f\x94\xb9': '🔹',  # Blue diamond  
            'â\x9a\xa0ï¸': '⚠️',    # Warning sign
            'â\x9c\x85': '✅',      # Check mark
            'ð\x9f\x92\xa1': '💡', # Light bulb
        }
        
        original_content = content
        changed = False
        
        for corrupt, correct in fixes.items():
            if corrupt in content:
                content = content.replace(corrupt, correct)
                changed = True
                print(f"  Fixed {corrupt} -> {correct}")
        
        # Save the file back if changes were made
        if changed:
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                f.write(content)
            print(f"✅ Fixed encoding in: {file_path}")
            return True
        else:
            print(f"  No encoding issues found in: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Fix encoding issues in all Python files"""
    print("🔧 Fixing Unicode encoding issues...")
    print("=" * 50)
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files to check...")
    print()
    
    fixed_count = 0
    for file_path in python_files:
        print(f"Checking: {file_path}")
        if fix_unicode_in_file(file_path):
            fixed_count += 1
        print()
    
    print("=" * 50)
    print(f"📊 Results: Fixed encoding in {fixed_count} files")
    
    if fixed_count > 0:
        print("🎉 Unicode encoding issues have been resolved!")
        print("Your application should now display emojis correctly.")
    else:
        print("ℹ️  No encoding issues found in the codebase.")

if __name__ == "__main__":
    main()
