#!/usr/bin/env python3
"""
update_data_loader.py - Update data_loader.py with the fix

This script updates your local data_loader.py with the fixed version
that handles MRI date columns correctly.
"""

import shutil
from pathlib import Path
import sys

def main():
    print("=" * 80)
    print("UPDATING data_loader.py with MRI date column fix")
    print("=" * 80)
    print()
    
    # Find the files
    fixed_file = Path('/mnt/user-data/outputs/data_loader.py')
    local_file = Path('data_loader.py')
    backup_file = Path('data_loader.py.backup')
    
    # Check if fixed file exists
    if not fixed_file.exists():
        print("❌ Error: Fixed data_loader.py not found in /mnt/user-data/outputs/")
        print("   You may need to copy it manually.")
        sys.exit(1)
    
    # Check if local file exists
    if not local_file.exists():
        print("❌ Error: data_loader.py not found in current directory")
        print("   Make sure you're in the tmvhnn directory")
        print(f"   Current directory: {Path.cwd()}")
        sys.exit(1)
    
    print("Found files:")
    print(f"  Fixed version: {fixed_file}")
    print(f"  Your version:  {local_file}")
    print()
    
    # Create backup
    print(f"Creating backup: {backup_file}")
    shutil.copy2(local_file, backup_file)
    print("✓ Backup created")
    print()
    
    # Copy fixed version
    print("Copying fixed version...")
    shutil.copy2(fixed_file, local_file)
    print("✓ Updated data_loader.py")
    print()
    
    print("=" * 80)
    print("✅ UPDATE COMPLETE")
    print("=" * 80)
    print()
    print("What was fixed:")
    print("  1. MRI date columns (like MRI_SCAN_DATE) are now excluded")
    print("  2. String values are skipped during feature extraction")
    print("  3. Better progress logging added")
    print()
    print("Your original file was backed up to: data_loader.py.backup")
    print()
    print("You can now run TMV-HNN successfully:")
    print("  python3 run.py --patient_path output/nacc_patient_table_*.csv ...")
    print()

if __name__ == "__main__":
    main()
