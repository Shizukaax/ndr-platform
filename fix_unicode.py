"""
Comprehensive encoding fix script - handles binary-level fixes
"""
import os

def fix_file_encoding(filepath):
    """Fix encoding issues in a file"""
    print(f"Fixing: {filepath}")
    
    # Read as binary first to see the actual bytes
    with open(filepath, 'rb') as f:
        data = f.read()
    
    # Convert to string with error handling
    try:
        content = data.decode('utf-8', errors='replace')
    except:
        content = data.decode('latin-1', errors='replace')
    
    # Define replacements for corrupted Unicode
    replacements = [
        # Corrupted blue diamond emoji
        ('ðŸ"¹', '🔹'),
        # Corrupted warning emoji 
        ('⚠️', '⚠️'),
        # Corrupted checkmark
        ('✅', '✅'),
        # Alternative patterns
        ('ð\x9f\x94\xb9', '🔹'),
        ('â\x9a\xa0', '⚠'),
        ('â\x9c\x85', '✅'),
    ]
    
    original_content = content
    changes_made = False
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changes_made = True
            print(f"  Replaced: {repr(old)} -> {repr(new)}")
    
    # Write back as UTF-8
    if changes_made:
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            f.write(content)
        print(f"  ✅ Saved {filepath}")
        return True
    else:
        print(f"  No changes needed for {filepath}")
        return False

def main():
    """Fix encoding in specific files"""
    files_to_fix = [
        'app/pages/real_time_monitoring.py',
        'app/pages/auto_labeling.py',
    ]
    
    fixed_count = 0
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            if fix_file_encoding(filepath):
                fixed_count += 1
        else:
            print(f"File not found: {filepath}")
    
    print(f"\n📊 Fixed encoding in {fixed_count} files")

if __name__ == "__main__":
    main()
