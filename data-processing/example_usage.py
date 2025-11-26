#!/usr/bin/env python3
"""
example_usage.py - Example script showing programmatic usage of NACC processor

This script demonstrates how to use the NACC dataset processor modules
directly in Python code without using the command-line interface.
"""

import pandas as pd
from nacc_processor import NACCDataProcessor
from data_quality import DataQualityChecker, validate_nacc_data
from utils import (
    save_datasets,
    create_summary_statistics,
    create_baseline_dataset,
    print_processing_summary
)

def example_basic_processing():
    """Example 1: Basic processing with UDS data only."""
    print("Example 1: Basic Processing")
    print("-" * 50)
    
    # Initialize processor
    processor = NACCDataProcessor(
        uds_path="data/uds_data.csv",
        mri_path=None,
        temporal_threshold=90
    )
    
    # Process datasets
    datasets = processor.process_datasets()
    
    # Access the normalized tables
    patient_df = datasets['patient']
    visit_df = datasets['visit']
    adc_df = datasets['adc']
    
    print(f"Patient table shape: {patient_df.shape}")
    print(f"Visit table shape: {visit_df.shape}")
    print(f"ADC table shape: {adc_df.shape}")
    
    # Save to CSV
    save_datasets(datasets, "output/basic/", format='csv')
    print("\nDatasets saved to output/basic/")
    

def example_with_mri():
    """Example 2: Processing with MRI data integration."""
    print("\nExample 2: Processing with MRI Integration")
    print("-" * 50)
    
    # Initialize processor with MRI data
    processor = NACCDataProcessor(
        uds_path="data/uds_data.csv",
        mri_path="data/scan_mri.csv",
        temporal_threshold=60  # 60-day matching window
    )
    
    # Process datasets
    datasets = processor.process_datasets()
    visit_df = datasets['visit']
    
    # Check MRI integration
    mri_cols = [col for col in visit_df.columns if col.startswith('MRI_')]
    print(f"Number of MRI columns added: {len(mri_cols)}")
    
    if mri_cols:
        visits_with_mri = visit_df[mri_cols[0]].notna().sum()
        print(f"Visits with MRI data: {visits_with_mri} ({visits_with_mri/len(visit_df)*100:.1f}%)")
        

def example_quality_analysis():
    """Example 3: Data quality analysis."""
    print("\nExample 3: Data Quality Analysis")
    print("-" * 50)
    
    # Process data first
    processor = NACCDataProcessor(uds_path="data/uds_data.csv")
    datasets = processor.process_datasets()
    
    # Run quality validation
    quality_report = validate_nacc_data(
        datasets['patient'],
        datasets['visit'],
        datasets['adc']
    )
    
    # Check validation results
    validations = quality_report['validations']
    print(f"Validation checks passed: {validations['passed'].sum()}/{len(validations)}")
    
    # Check missing data patterns
    patient_missing = quality_report['patient_missing']
    high_missing = patient_missing[patient_missing['null_percentage'] > 50]
    if len(high_missing) > 0:
        print(f"\nPatient variables with >50% missing:")
        for _, row in high_missing.iterrows():
            print(f"  - {row['column']}: {row['null_percentage']:.1f}%")
            

def example_custom_analysis():
    """Example 4: Custom analysis with the processed data."""
    print("\nExample 4: Custom Analysis")
    print("-" * 50)
    
    # Process data
    processor = NACCDataProcessor(uds_path="data/uds_data.csv")
    datasets = processor.process_datasets()
    
    patient_df = datasets['patient']
    visit_df = datasets['visit']
    
    # Example: APOE e4 carrier analysis
    if 'NACCAPOE' in patient_df.columns:
        apoe_e4_carriers = patient_df['NACCAPOE'].isin([24, 34, 44])
        print(f"APOE e4 carriers: {apoe_e4_carriers.sum()} ({apoe_e4_carriers.mean()*100:.1f}%)")
    
    # Example: Visit frequency by year
    if 'VISITYR' in visit_df.columns:
        visits_by_year = visit_df.groupby('VISITYR').size()
        print(f"\nVisits by year (last 5 years):")
        for year, count in visits_by_year.tail().items():
            print(f"  {year}: {count:,} visits")
            
    # Create and analyze baseline dataset
    baseline_df = create_baseline_dataset(patient_df, visit_df)
    print(f"\nBaseline dataset shape: {baseline_df.shape}")
    

def example_statistical_export():
    """Example 5: Export for statistical software."""
    print("\nExample 5: Statistical Software Export")
    print("-" * 50)
    
    # Process data
    processor = NACCDataProcessor(uds_path="data/uds_data.csv")
    datasets = processor.process_datasets()
    
    # Export in different formats
    # Parquet format (efficient for large datasets)
    save_datasets(datasets, "output/parquet/", format='parquet', compression='snappy')
    print("Saved in Parquet format (good for Python/R)")
    
    # Stata format
    save_datasets(datasets, "output/stata/", format='stata')
    print("Saved in Stata format")
    
    # Excel format (all tables in one file)
    save_datasets(datasets, "output/excel/", format='excel')
    print("Saved in Excel format")
    

def main():
    """Run all examples."""
    print("NACC Dataset Processor - Usage Examples")
    print("=" * 50)
    
    # Note: These examples assume you have data files in the specified paths
    # Adjust the paths according to your data location
    
    try:
        example_basic_processing()
    except FileNotFoundError:
        print("Note: Could not find data files for Example 1")
        
    try:
        example_with_mri()
    except FileNotFoundError:
        print("Note: Could not find data files for Example 2")
        
    try:
        example_quality_analysis()
    except FileNotFoundError:
        print("Note: Could not find data files for Example 3")
        
    try:
        example_custom_analysis()
    except FileNotFoundError:
        print("Note: Could not find data files for Example 4")
        
    try:
        example_statistical_export()
    except FileNotFoundError:
        print("Note: Could not find data files for Example 5")
        
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNote: Adjust file paths in the examples to match your data location.")
    

if __name__ == "__main__":
    main()