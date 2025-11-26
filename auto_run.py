#!/usr/bin/env python3
"""
auto_run.py - Automatically find and use the latest NACC files

This script searches for NACC data files in the output directory
and automatically runs TMV-HNN with the most recent files.
"""

import sys
import os
from pathlib import Path
import subprocess
import glob
from datetime import datetime

def find_latest_nacc_files():
    """Find the most recent NACC data files."""
    
    # Search in common locations
    search_dirs = [
        Path('output'),
        Path('.'),
        Path('..') / 'output',
        Path('../nacc_processor/output') if Path('../nacc_processor').exists() else None
    ]
    
    # Filter out None values
    search_dirs = [d for d in search_dirs if d is not None and d.exists()]
    
    patient_files = []
    visit_files = []
    adc_files = []
    
    for search_dir in search_dirs:
        patient_files.extend(search_dir.glob('nacc_patient_table_*.csv'))
        visit_files.extend(search_dir.glob('nacc_visit_table_*.csv'))
        adc_files.extend(search_dir.glob('nacc_adc_table_*.csv'))
    
    if not patient_files or not visit_files:
        return None, None, None
    
    # Get the most recent files (by modification time)
    patient_file = max(patient_files, key=lambda p: p.stat().st_mtime)
    visit_file = max(visit_files, key=lambda p: p.stat().st_mtime)
    adc_file = max(adc_files, key=lambda p: p.stat().st_mtime) if adc_files else None
    
    return patient_file, visit_file, adc_file


def main():
    """Main function."""
    print("=" * 80)
    print("TMV-HNN Auto-Run: Finding NACC Data Files")
    print("=" * 80)
    print()
    
    # Find files
    print("Searching for NACC data files...")
    patient_file, visit_file, adc_file = find_latest_nacc_files()
    
    if not patient_file or not visit_file:
        print("\n❌ ERROR: Could not find NACC data files!")
        print("\nPlease ensure you have processed NACC files in one of these locations:")
        print("  - ./output/")
        print("  - ../output/")
        print("  - ../nacc_processor/output/")
        print("\nOr run with explicit paths:")
        print("  python3 run.py --patient_path PATH --visit_path PATH")
        sys.exit(1)
    
    # Display found files
    print("\n✓ Found NACC data files:")
    print(f"  Patient table: {patient_file}")
    print(f"  Visit table:   {visit_file}")
    if adc_file:
        print(f"  ADC table:     {adc_file}")
    else:
        print(f"  ADC table:     Not found (will skip)")
    
    # Check file sizes
    print("\nFile sizes:")
    print(f"  Patient table: {patient_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Visit table:   {visit_file.stat().st_size / 1024 / 1024:.1f} MB")
    if adc_file:
        print(f"  ADC table:     {adc_file.stat().st_size / 1024:.1f} KB")
    
    # Verify file structure
    print("\nVerifying data structure...")
    try:
        import pandas as pd
        
        # Quick check of patient table
        patient_df = pd.read_csv(patient_file, nrows=5)
        required_cols = ['NACCID', 'SEX', 'BIRTHYR']
        missing_cols = [col for col in required_cols if col not in patient_df.columns]
        
        if missing_cols:
            print(f"⚠️  WARNING: Patient table missing columns: {missing_cols}")
            print("   This might not be a properly formatted NACC patient table.")
        else:
            print("✓ Patient table structure looks good")
        
        # Quick check of visit table
        visit_df = pd.read_csv(visit_file, nrows=5)
        required_cols = ['NACCID', 'NACCVNUM', 'VISITYR']
        missing_cols = [col for col in required_cols if col not in visit_df.columns]
        
        if missing_cols:
            print(f"⚠️  WARNING: Visit table missing columns: {missing_cols}")
            print("   This might not be a properly formatted NACC visit table.")
        else:
            print("✓ Visit table structure looks good")
            
    except Exception as e:
        print(f"⚠️  Could not verify file structure: {e}")
    
    # Build command
    cmd = [
        'python3', 'run.py',
        '--patient_path', str(patient_file),
        '--visit_path', str(visit_file),
    ]
    
    if adc_file:
        cmd.extend(['--adc_path', str(adc_file)])
    
    # Add any additional arguments passed to this script
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    print("\n" + "=" * 80)
    print("Running TMV-HNN with detected files:")
    print("=" * 80)
    print("\nCommand:")
    print(" ".join(cmd))
    print()
    
    # Ask for confirmation
    response = input("Proceed with training? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Run the command
    print("\n" + "=" * 80)
    print("STARTING TMV-HNN TRAINING")
    print("=" * 80)
    print()
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running TMV-HNN: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
