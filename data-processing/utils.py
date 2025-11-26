# utils.py
"""
Utility functions for NACC dataset processing.
Includes data export, summary statistics, and helper functions.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
from datetime import datetime

logger = logging.getLogger(__name__)


def save_datasets(datasets: Dict[str, pd.DataFrame], output_dir: str, 
                 format: str = 'csv', compression: Optional[str] = None) -> None:
    """
    Save normalized datasets to files.
    
    Args:
        datasets: Dictionary with 'patient', 'visit', and 'adc' DataFrames
        output_dir: Directory to save files
        format: Output format ('csv', 'parquet', 'stata', 'excel')
        compression: Optional compression ('gzip', 'bz2', 'zip', 'xz')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving datasets to {output_path} in {format} format...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for name, df in datasets.items():
        if format == 'csv':
            filename = f"nacc_{name}_table_{timestamp}.csv"
            if compression:
                filename += f".{compression}"
            filepath = output_path / filename
            df.to_csv(filepath, index=False, compression=compression)
            
        elif format == 'parquet':
            filename = f"nacc_{name}_table_{timestamp}.parquet"
            filepath = output_path / filename
            df.to_parquet(filepath, index=False, compression=compression)
            
        elif format == 'stata':
            filename = f"nacc_{name}_table_{timestamp}.dta"
            filepath = output_path / filename
            # Stata has variable name limitations
            df_stata = df.copy()
            df_stata.columns = [col[:32] for col in df_stata.columns]
            df_stata.to_stata(filepath, write_index=False)
            
        elif format == 'excel':
            filename = f"nacc_normalized_tables_{timestamp}.xlsx"
            filepath = output_path / filename
            with pd.ExcelWriter(filepath) as writer:
                for sheet_name, sheet_df in datasets.items():
                    sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"Saved all tables to {filepath}")
            return
            
        logger.info(f"Saved {name} table to {filepath}")
        
        
def generate_codebook(datasets: Dict[str, pd.DataFrame], output_path: str) -> None:
    """
    Generate a codebook documenting all variables in the normalized datasets.
    
    Args:
        datasets: Dictionary with 'patient', 'visit', and 'adc' DataFrames
        output_path: Path to save codebook
    """
    logger.info("Generating codebook...")
    
    codebook = []
    
    for table_name, df in datasets.items():
        for col in df.columns:
            entry = {
                'table': table_name,
                'variable': col,
                'type': str(df[col].dtype),
                'non_null_count': df[col].notna().sum(),
                'null_count': df[col].isna().sum(),
                'unique_values': df[col].nunique()
            }
            
            # Add descriptive statistics for numeric variables
            if df[col].dtype in ['int64', 'float64']:
                entry['mean'] = df[col].mean()
                entry['std'] = df[col].std()
                entry['min'] = df[col].min()
                entry['25%'] = df[col].quantile(0.25)
                entry['50%'] = df[col].quantile(0.50)
                entry['75%'] = df[col].quantile(0.75)
                entry['max'] = df[col].max()
                
            # Sample unique values for categorical variables
            elif df[col].dtype == 'object':
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 20:
                    entry['unique_values_list'] = list(unique_vals)
                else:
                    entry['unique_values_sample'] = list(unique_vals[:20])
                    
            codebook.append(entry)
            
    codebook_df = pd.DataFrame(codebook)
    codebook_df.to_csv(output_path, index=False)
    logger.info(f"Saved codebook to {output_path}")
    
    
def create_summary_statistics(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Create summary statistics for the normalized datasets.
    
    Args:
        datasets: Dictionary with 'patient', 'visit', and 'adc' DataFrames
        
    Returns:
        Dictionary with summary statistics DataFrames
    """
    logger.info("Creating summary statistics...")
    
    summary_stats = {}
    
    # Patient demographics summary
    if 'patient' in datasets:
        patient_df = datasets['patient']
        
        demographics = pd.DataFrame({
            'Total Patients': [len(patient_df)],
            'Female (%)': [(patient_df['SEX'] == 2).sum() / len(patient_df) * 100] if 'SEX' in patient_df else [np.nan],
            'Mean Age at Baseline': [patient_df['NACCAGEB'].mean()] if 'NACCAGEB' in patient_df else [np.nan],
            'APOE e4+ (%)': [(patient_df['NACCAPOE'].isin([24, 34, 44])).sum() / patient_df['NACCAPOE'].notna().sum() * 100] if 'NACCAPOE' in patient_df else [np.nan],
            'Deceased (%)': [(patient_df['NACCDIED'] == 1).sum() / len(patient_df) * 100] if 'NACCDIED' in patient_df else [np.nan]
        })
        summary_stats['demographics'] = demographics
        
        # Visit distribution
        if 'N_VISITS' in patient_df:
            visit_dist = patient_df['N_VISITS'].value_counts().sort_index()
            summary_stats['visit_distribution'] = visit_dist.to_frame('count')
            
    # Visit summary
    if 'visit' in datasets:
        visit_df = datasets['visit']
        
        visit_summary = pd.DataFrame({
            'Total Visits': [len(visit_df)],
            'Unique Patients': [visit_df['NACCID'].nunique()],
            'Year Range': [f"{visit_df['VISITYR'].min()}-{visit_df['VISITYR'].max()}"] if 'VISITYR' in visit_df else ['Unknown']
        })
        
        # MRI availability
        mri_cols = [col for col in visit_df.columns if col.startswith('MRI_')]
        if mri_cols:
            mri_available = visit_df[mri_cols[0]].notna().sum()
            visit_summary['Visits with MRI (%)'] = [mri_available / len(visit_df) * 100]
            
        summary_stats['visit_summary'] = visit_summary
        
    # ADC summary
    if 'adc' in datasets:
        adc_df = datasets['adc']
        
        adc_summary = adc_df[['N_VISITS', 'N_PATIENTS']].describe()
        summary_stats['adc_summary'] = adc_summary
        
    return summary_stats
    

def create_baseline_dataset(patient_df: pd.DataFrame, visit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a baseline-only dataset by merging patient data with first visits.
    
    Args:
        patient_df: Patient-level DataFrame
        visit_df: Visit-level DataFrame
        
    Returns:
        Baseline dataset with one row per patient
    """
    logger.info("Creating baseline dataset...")
    
    # Get first visits only (NACCVNUM == 1)
    baseline_visits = visit_df[visit_df['NACCVNUM'] == 1].copy()
    
    # Merge with patient data
    baseline_df = patient_df.merge(
        baseline_visits,
        on='NACCID',
        how='left',
        suffixes=('_patient', '_visit')
    )
    
    # Handle duplicate columns (prefer visit version for time-varying variables)
    for col in baseline_df.columns:
        if col.endswith('_visit'):
            base_col = col[:-6]  # Remove '_visit' suffix
            if f'{base_col}_patient' in baseline_df.columns:
                # Use visit version and drop patient version
                baseline_df[base_col] = baseline_df[col]
                baseline_df = baseline_df.drop(columns=[col, f'{base_col}_patient'])
            else:
                # Rename to remove suffix
                baseline_df = baseline_df.rename(columns={col: base_col})
        elif col.endswith('_patient'):
            # Rename to remove suffix if no visit version exists
            base_col = col[:-8]  # Remove '_patient' suffix
            if base_col not in baseline_df.columns:
                baseline_df = baseline_df.rename(columns={col: base_col})
                
    logger.info(f"Created baseline dataset: {len(baseline_df)} patients")
    
    return baseline_df
    

def export_for_statistical_software(datasets: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """
    Export datasets with proper formatting for common statistical software.
    
    Args:
        datasets: Dictionary with normalized DataFrames
        output_dir: Directory to save exported files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # R-friendly export
    logger.info("Exporting for R...")
    for name, df in datasets.items():
        # Replace spaces in column names with underscores
        df_r = df.copy()
        df_r.columns = df_r.columns.str.replace(' ', '_').str.replace('-', '_')
        
        # Save as CSV for R
        df_r.to_csv(output_path / f"{name}_for_r.csv", index=False)
        
    # SAS-friendly export (column names ≤ 8 characters)
    logger.info("Exporting for SAS...")
    for name, df in datasets.items():
        df_sas = df.copy()
        # Truncate column names to 8 characters
        df_sas.columns = [col[:8] for col in df_sas.columns]
        
        # Handle duplicate column names
        seen = {}
        new_cols = []
        for col in df_sas.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col[:6]}{seen[col]}")
            else:
                seen[col] = 1
                new_cols.append(col)
        df_sas.columns = new_cols
        
        # Save as CSV for SAS
        df_sas.to_csv(output_path / f"{name}_for_sas.csv", index=False)
        
    logger.info(f"Exported datasets for statistical software to {output_path}")
    

def print_processing_summary(datasets: Dict[str, pd.DataFrame], 
                           quality_report: Optional[Dict[str, pd.DataFrame]] = None) -> None:
    """
    Print a comprehensive summary of the processing results.
    
    Args:
        datasets: Dictionary with normalized DataFrames
        quality_report: Optional quality report from data validation
    """
    print("\n" + "="*80)
    print("NACC DATASET PROCESSING SUMMARY")
    print("="*80)
    
    # Dataset dimensions
    print("\nDataset Dimensions:")
    for name, df in datasets.items():
        print(f"  {name.capitalize()} table: {len(df):,} rows × {len(df.columns)} columns")
        
    # Key statistics
    if 'patient' in datasets:
        patient_df = datasets['patient']
        print(f"\nPatient Statistics:")
        print(f"  Total unique patients: {len(patient_df):,}")
        
        if 'SEX' in patient_df:
            female_pct = (patient_df['SEX'] == 2).sum() / len(patient_df) * 100
            print(f"  Female: {female_pct:.1f}%")
            
        if 'NACCAPOE' in patient_df:
            apoe_available = patient_df['NACCAPOE'].notna().sum()
            print(f"  APOE genotype available: {apoe_available:,} ({apoe_available/len(patient_df)*100:.1f}%)")
            
    if 'visit' in datasets:
        visit_df = datasets['visit']
        print(f"\nVisit Statistics:")
        print(f"  Total visits: {len(visit_df):,}")
        
        mri_cols = [col for col in visit_df.columns if col.startswith('MRI_')]
        if mri_cols:
            mri_available = visit_df[mri_cols[0]].notna().sum()
            print(f"  Visits with MRI: {mri_available:,} ({mri_available/len(visit_df)*100:.1f}%)")
            
    # Data quality summary
    if quality_report and 'validations' in quality_report:
        print("\nData Quality Validation:")
        validations = quality_report['validations']
        passed = validations['passed'].sum()
        total = len(validations)
        print(f"  Passed {passed}/{total} validation checks")
        
        failed = validations[~validations['passed']]
        if len(failed) > 0:
            print("  Failed checks:")
            for _, row in failed.iterrows():
                print(f"    - {row['check']}")
                
    print("\n" + "="*80)