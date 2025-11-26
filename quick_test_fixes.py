#!/usr/bin/env python3
"""
quick_test_fixes.py - Quick test to validate TMV-HNN fixes

This script runs a quick test (1000 patients, 10 epochs) to verify that
the three critical fixes are working correctly.

Usage:
    python3 quick_test_fixes.py
"""

import sys
import subprocess
from pathlib import Path
import time

def print_header(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)

def main():
    print_header("TMV-HNN FIXES VALIDATION TEST")
    print("\nThis quick test will:")
    print("  1. Run TMV-HNN on 1,000 patients with 10 epochs per layer")
    print("  2. Check that all three fixes are working correctly")
    print("  3. Generate a validation report")
    print("\nEstimated time: 10-15 minutes")
    
    # Ask for confirmation
    response = input("\nProceed with test? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Test cancelled.")
        sys.exit(0)
    
    print_header("RUNNING QUICK TEST")
    
    # Build command
    cmd = [
        'python3', 'run.py',
        '--max_patients', '1000',
        '--epochs_l3', '10',
        '--epochs_l2', '10',
        '--epochs_l1', '10',
        '--epochs_survival', '10',
        '--output_dir', 'quick_test_results/'
    ]
    
    # Find data files
    data_dir = Path('output')
    if not data_dir.exists():
        data_dir = Path('.')
    
    patient_files = list(data_dir.glob('*patient*.csv'))
    visit_files = list(data_dir.glob('*visit*.csv'))
    
    if patient_files and visit_files:
        cmd.extend(['--patient_path', str(patient_files[0])])
        cmd.extend(['--visit_path', str(visit_files[0])])
        print(f"\nUsing data files:")
        print(f"  Patient: {patient_files[0]}")
        print(f"  Visit: {visit_files[0]}")
    
    print(f"\nRunning: {' '.join(cmd)}")
    print()
    
    # Run test
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        success = True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with error code {e.returncode}")
        success = False
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        success = False
    
    end_time = time.time()
    duration = end_time - start_time
    
    print_header("TEST COMPLETED")
    print(f"\nDuration: {duration/60:.1f} minutes")
    
    if not success:
        print("\n❌ Test did not complete successfully")
        print("Check the log file for errors:")
        print("  tmv_hnn_*.log")
        sys.exit(1)
    
    # Analyze results
    print_header("VALIDATING RESULTS")
    
    results_dir = Path('quick_test_results')
    
    if not results_dir.exists():
        print("\n❌ Results directory not found!")
        sys.exit(1)
    
    # Check for output files
    expected_files = [
        'patient_clusters.csv',
        'patient_embeddings.csv',
        'survival_predictions.csv',
        'summary.json'
    ]
    
    print("\n✓ Checking output files:")
    all_present = True
    for fname in expected_files:
        if (results_dir / fname).exists():
            print(f"  ✓ {fname}")
        else:
            print(f"  ❌ {fname} - MISSING")
            all_present = False
    
    if not all_present:
        print("\n❌ Some output files are missing!")
        sys.exit(1)
    
    # Validate cluster balance
    print("\n✓ Validating Layer 3 clustering:")
    try:
        import pandas as pd
        clusters = pd.read_csv(results_dir / 'patient_clusters.csv')
        cluster_counts = clusters['cluster_l3'].value_counts()
        max_cluster_pct = cluster_counts.max() / len(clusters) * 100
        
        print(f"  Total patients: {len(clusters)}")
        print(f"  Largest cluster: {max_cluster_pct:.1f}%")
        
        if max_cluster_pct > 80:
            print(f"  ❌ FAIL: Clustering still degenerate ({max_cluster_pct:.1f}% in one cluster)")
            print("  → Fix 1 may not be working correctly")
        elif max_cluster_pct > 40:
            print(f"  ⚠️ WARNING: Largest cluster has {max_cluster_pct:.1f}% of patients")
            print("  → Consider adjusting adaptive_sigma scale factor")
        else:
            print(f"  ✓ PASS: Clustering is reasonably balanced")
    except Exception as e:
        print(f"  ⚠️ Could not validate clustering: {e}")
    
    # Validate survival predictions
    print("\n✓ Validating survival predictions:")
    try:
        survival = pd.read_csv(results_dir / 'survival_predictions.csv', index_col=0)
        
        # Check 1-year predictions
        one_year = survival['1yr']
        
        print(f"  1-year survival mean: {one_year.mean():.3f}")
        print(f"  1-year survival range: [{one_year.min():.3f}, {one_year.max():.3f}]")
        
        if one_year.mean() < 0.01:
            print(f"  ❌ FAIL: Survival predictions too low (mean={one_year.mean():.3f})")
            print("  → Fix 3 (hazard clipping or integration) may not be working")
        elif one_year.mean() > 0.99:
            print(f"  ❌ FAIL: Survival predictions too high (mean={one_year.mean():.3f})")
            print("  → Model may not be learning")
        elif one_year.min() < 0.01 or one_year.max() > 0.99:
            print(f"  ⚠️ WARNING: Some predictions are extreme")
        else:
            print(f"  ✓ PASS: Survival predictions in reasonable range")
    except Exception as e:
        print(f"  ⚠️ Could not validate survival predictions: {e}")
    
    # Check log file for key metrics
    print("\n✓ Checking log file:")
    try:
        import glob
        log_files = glob.glob('tmv_hnn_*.log')
        if not log_files:
            print("  ⚠️ No log file found")
        else:
            latest_log = max(log_files, key=lambda f: Path(f).stat().st_mtime)
            with open(latest_log, 'r') as f:
                log_content = f.read()
            
            # Check for critical issues
            issues_found = []
            
            if "Loss: 0.0000" in log_content and "Layer 2" in log_content:
                issues_found.append("Layer 2 zero loss detected")
            
            if "C-index: 0." in log_content:
                # Extract C-index value
                import re
                c_index_match = re.search(r'C-index: (0\.\d+)', log_content)
                if c_index_match:
                    c_index = float(c_index_match.group(1))
                    print(f"  C-index: {c_index:.4f}")
                    
                    if c_index < 0.4:
                        issues_found.append(f"C-index extremely low ({c_index:.4f})")
                    elif c_index < 0.5:
                        issues_found.append(f"C-index worse than random ({c_index:.4f})")
                    elif c_index > 0.55:
                        print(f"  ✓ C-index is better than random!")
            
            if issues_found:
                print(f"  ⚠️ Issues found in log:")
                for issue in issues_found:
                    print(f"    - {issue}")
            else:
                print(f"  ✓ No critical issues detected")
    except Exception as e:
        print(f"  ⚠️ Could not check log file: {e}")
    
    print_header("VALIDATION COMPLETE")
    
    if all_present and max_cluster_pct < 80 and one_year.mean() > 0.1:
        print("\n✅ ALL CHECKS PASSED!")
        print("\nThe fixes appear to be working correctly.")
        print("You can now run the full training with:")
        print("\n  python3 run.py --max_patients 5000 --epochs_l3 100 --epochs_l2 100 --epochs_l1 100 --epochs_survival 100")
        sys.exit(0)
    else:
        print("\n⚠️ SOME ISSUES DETECTED")
        print("\nPlease review the validation results above.")
        print("See FIXES_SUMMARY.md for troubleshooting guidance.")
        sys.exit(1)


if __name__ == "__main__":
    main()
