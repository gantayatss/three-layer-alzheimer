#!/usr/bin/env python3
"""
find_nacc_data.py - Helper script to locate NACC data files

This script searches for NACC patient/visit/adc table files in common locations.
"""

import os
import sys
from pathlib import Path
import glob

def find_nacc_files():
    """Search for NACC data files in common locations."""
    
    print("Searching for NACC data files...")
    print("=" * 80)
    
    # Common search locations
    search_paths = [
        Path.cwd(),  # Current directory
        Path.cwd() / "output",
        Path.cwd() / "results",
        Path.cwd() / "data",
        Path.cwd().parent / "output",
        Path.home() / "Documents" / "Research",
    ]
    
    found_files = {
        'patient': [],
        'visit': [],
        'adc': []
    }
    
    # Search patterns
    patterns = {
        'patient': ['*patient*.csv', '*PATIENT*.csv'],
        'visit': ['*visit*.csv', '*VISIT*.csv'],
        'adc': ['*adc*.csv', '*ADC*.csv']
    }
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
            
        print(f"\nSearching in: {search_path}")
        
        for file_type, file_patterns in patterns.items():
            for pattern in file_patterns:
                matches = list(search_path.glob(pattern))
                if matches:
                    found_files[file_type].extend(matches)
                    
    # Print results
    print("\n" + "=" * 80)
    print("FOUND FILES:")
    print("=" * 80)
    
    if found_files['patient']:
        print("\n📊 Patient table files:")
        for f in found_files['patient']:
            print(f"  ✓ {f}")
            
    if found_files['visit']:
        print("\n📊 Visit table files:")
        for f in found_files['visit']:
            print(f"  ✓ {f}")
            
    if found_files['adc']:
        print("\n📊 ADC table files:")
        for f in found_files['adc']:
            print(f"  ✓ {f}")
            
    # Generate command
    if found_files['patient'] and found_files['visit']:
        print("\n" + "=" * 80)
        print("✅ READY TO RUN TMV-HNN!")
        print("=" * 80)
        print("\nUse this command:\n")
        
        patient_file = str(found_files['patient'][0])
        visit_file = str(found_files['visit'][0])
        
        cmd = f"python3 run.py \\\n    --patient_path '{patient_file}' \\\n    --visit_path '{visit_file}'"
        
        if found_files['adc']:
            adc_file = str(found_files['adc'][0])
            cmd += f" \\\n    --adc_path '{adc_file}'"
            
        print(cmd)
        print()
        
    else:
        print("\n" + "=" * 80)
        print("❌ NO NACC DATA FILES FOUND")
        print("=" * 80)
        print("\nYou need to generate NACC data files first.")
        print("\nOption 1: If you have raw NACC data (investigator_nacc*.csv files):")
        print("  1. Make sure you have nacc_processor.py in your project")
        print("  2. Run: python3 run.py --uds_path data/investigator_nacc69.csv --mri_path data/investigator_scan_mrisbm_nacc69.csv")
        print("\nOption 2: If you have the NACC data elsewhere:")
        print("  1. Find your patient/visit table CSV files")
        print("  2. Run: python3 run.py --patient_path PATH_TO_PATIENT.csv --visit_path PATH_TO_VISIT.csv")
        print()

if __name__ == "__main__":
    find_nacc_files()
