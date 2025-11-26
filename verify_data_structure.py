#!/usr/bin/env python3
"""
verify_data_structure.py - Verify NACC data structure compatibility

Checks if your NACC data files have the expected structure for TMV-HNN.
"""

import pandas as pd
import sys
from pathlib import Path

def check_patient_table(file_path):
    """Check patient table structure."""
    print("\n" + "="*80)
    print("CHECKING PATIENT TABLE")
    print("="*80)
    
    try:
        df = pd.read_csv(file_path, nrows=100)
        print(f"✓ Successfully loaded: {file_path}")
        print(f"  Rows (sample): {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        
        # Expected columns
        expected_cols = {
            'Demographics': ['NACCID', 'SEX', 'BIRTHYR', 'RACE', 'HISPANIC', 'EDUC'],
            'Genetics': ['NACCAPOE', 'NACCNE4S'],
            'Visit Stats': ['N_VISITS', 'FIRST_VISITYR', 'LAST_VISITYR'],
            'Death Info': ['NACCDIED']
        }
        
        all_good = True
        for category, cols in expected_cols.items():
            found = [col for col in cols if col in df.columns]
            missing = [col for col in cols if col not in df.columns]
            
            print(f"\n{category}:")
            if found:
                print(f"  ✓ Found: {', '.join(found)}")
            if missing:
                print(f"  ⚠️  Missing: {', '.join(missing)}")
                all_good = False
        
        # Sample data
        print(f"\nSample data (first 3 rows):")
        print(df[['NACCID', 'SEX', 'BIRTHYR']].head(3).to_string())
        
        return all_good
        
    except Exception as e:
        print(f"❌ Error reading patient table: {e}")
        return False


def check_visit_table(file_path):
    """Check visit table structure."""
    print("\n" + "="*80)
    print("CHECKING VISIT TABLE")
    print("="*80)
    
    try:
        df = pd.read_csv(file_path, nrows=100)
        print(f"✓ Successfully loaded: {file_path}")
        print(f"  Rows (sample): {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        
        # Expected columns
        expected_cols = {
            'Visit IDs': ['NACCID', 'NACCVNUM', 'VISITYR', 'VISITMO'],
            'Cognitive Tests': ['MMSE', 'MOCA', 'LOGIMEM', 'TRAILA', 'TRAILB'],
            'Symptoms': ['DELUSION', 'HALLUCIN', 'AGIT', 'DEPD', 'ANX', 'APATHY'],
            'Functional': ['BILLS', 'TAXES', 'SHOPPING'],
            'Diagnosis': ['NORMCOG', 'NACCMCI', 'NACCALZD', 'DEMENTED']
        }
        
        all_good = True
        for category, cols in expected_cols.items():
            found = [col for col in cols if col in df.columns]
            missing = [col for col in cols if col not in df.columns]
            
            print(f"\n{category}:")
            if found:
                print(f"  ✓ Found: {', '.join(found)}")
            if missing:
                print(f"  ⚠️  Missing: {', '.join(missing)}")
                if category in ['Visit IDs', 'Cognitive Tests']:
                    all_good = False
        
        # Check for MRI data
        mri_cols = [col for col in df.columns if col.startswith('MRI_')]
        if mri_cols:
            print(f"\n✓ MRI Data Found: {len(mri_cols)} MRI columns")
            print(f"  Examples: {', '.join(mri_cols[:5])}")
        else:
            print(f"\n⚠️  No MRI data found (columns starting with 'MRI_')")
        
        # Sample data
        print(f"\nSample data (first 3 rows):")
        sample_cols = ['NACCID', 'NACCVNUM', 'VISITYR']
        sample_cols = [c for c in sample_cols if c in df.columns]
        if sample_cols:
            print(df[sample_cols].head(3).to_string())
        
        return all_good
        
    except Exception as e:
        print(f"❌ Error reading visit table: {e}")
        return False


def check_adc_table(file_path):
    """Check ADC table structure."""
    print("\n" + "="*80)
    print("CHECKING ADC TABLE")
    print("="*80)
    
    if not file_path or not Path(file_path).exists():
        print("⚠️  ADC table not provided or not found (optional)")
        return True
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Successfully loaded: {file_path}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        
        # Expected columns
        expected_cols = ['NACCADC', 'N_VISITS', 'N_PATIENTS']
        found = [col for col in expected_cols if col in df.columns]
        missing = [col for col in expected_cols if col not in df.columns]
        
        if found:
            print(f"  ✓ Found: {', '.join(found)}")
        if missing:
            print(f"  ⚠️  Missing: {', '.join(missing)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading ADC table: {e}")
        return False


def main():
    """Main function."""
    if len(sys.argv) < 3:
        print("Usage: python3 verify_data_structure.py PATIENT_FILE VISIT_FILE [ADC_FILE]")
        print("\nExample:")
        print("  python3 verify_data_structure.py \\")
        print("      output/nacc_patient_table_20251112_183501.csv \\")
        print("      output/nacc_visit_table_20251112_183501.csv \\")
        print("      output/nacc_adc_table_20251112_183501.csv")
        sys.exit(1)
    
    patient_file = sys.argv[1]
    visit_file = sys.argv[2]
    adc_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    print("=" * 80)
    print("TMV-HNN DATA STRUCTURE VERIFICATION")
    print("=" * 80)
    print(f"\nPatient file: {patient_file}")
    print(f"Visit file:   {visit_file}")
    if adc_file:
        print(f"ADC file:     {adc_file}")
    
    # Check each table
    patient_ok = check_patient_table(patient_file)
    visit_ok = check_visit_table(visit_file)
    adc_ok = check_adc_table(adc_file)
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    if patient_ok and visit_ok and adc_ok:
        print("\n✅ ALL CHECKS PASSED!")
        print("\nYour data structure is compatible with TMV-HNN.")
        print("\nYou can now run:")
        print(f"  python3 run.py \\")
        print(f"      --patient_path {patient_file} \\")
        print(f"      --visit_path {visit_file}")
        if adc_file:
            print(f"      --adc_path {adc_file}")
        sys.exit(0)
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\nYour data might be missing some expected columns.")
        print("TMV-HNN might still work, but some features may be unavailable.")
        print("\nTo proceed anyway, run:")
        print(f"  python3 run.py \\")
        print(f"      --patient_path {patient_file} \\")
        print(f"      --visit_path {visit_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()
