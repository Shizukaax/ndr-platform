# -*- coding: utf-8 -*-
"""
Simple script to fix Unicode encoding issues
"""
import os

def fix_real_time_monitoring():
    file_path = 'app/pages/real_time_monitoring.py'
    
    # Read with error replacement
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Simple replacements
    content = content.replace('ðŸ"¹', '🔹')
    content = content.replace('ð\x9f\x94\xb9', '🔹')
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed: {file_path}")

if __name__ == "__main__":
    fix_real_time_monitoring()
    print("Done!")
