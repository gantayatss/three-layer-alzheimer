# data_quality.py
"""
Data Quality and Validation Module
Handles missing data patterns, quality assessment, and validation for NACC datasets.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """
    Performs data quality checks and handles missing data patterns according to NACC conventions.
    """
    
    def __init__(self):
        """Initialize the data quality checker with NACC missing data codes."""
        self.missing_codes = {
            -4: "Not available: UDS form submitted did not collect data this way",
            88: "Not applicable",
            99: "Unknown or not obtained"
        }
        
    def check_missing_patterns(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        Analyze missing data patterns in a DataFrame.
        
        Args:
            df: DataFrame to analyze
            table_name: Name of the table for reporting
            
        Returns:
            DataFrame with missing data statistics
        """
        logger.info(f"Analyzing missing data patterns for {table_name} table...")
        
        missing_stats = []
        
        for col in df.columns:
            stats = {
                'column': col,
                'total_rows': len(df),
                'non_null_count': df[col].notna().sum(),
                'null_count': df[col].isna().sum(),
                'null_percentage': (df[col].isna().sum() / len(df)) * 100
            }
            
            # Check for NACC missing codes
            if df[col].dtype in ['int64', 'float64']:
                for code, description in self.missing_codes.items():
                    code_count = (df[col] == code).sum()
                    if code_count > 0:
                        stats[f'code_{code}_count'] = code_count
                        stats[f'code_{code}_desc'] = description
                        
            missing_stats.append(stats)
            
        missing_df = pd.DataFrame(missing_stats)
        missing_df = missing_df.sort_values('null_percentage', ascending=False)
        
        # Log summary
        high_missing = missing_df[missing_df['null_percentage'] > 50]
        if len(high_missing) > 0:
            logger.warning(f"{table_name}: {len(high_missing)} columns have >50% missing data")
            
        return missing_df
        
    def validate_relationships(self, patient_df: pd.DataFrame, visit_df: pd.DataFrame, 
                             adc_df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate relationships between the three normalized tables.
        
        Args:
            patient_df: Patient-level DataFrame
            visit_df: Visit-level DataFrame
            adc_df: ADC-level DataFrame
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating table relationships...")
        
        validations = {}
        
        # Check primary keys
        validations['patient_pk_unique'] = patient_df['NACCID'].is_unique
        validations['visit_pk_unique'] = not visit_df[['NACCID', 'NACCVNUM']].duplicated().any()
        validations['adc_pk_unique'] = adc_df['NACCADC'].is_unique
        
        # Check foreign key relationships
        # All visits should have a corresponding patient
        visit_patients = set(visit_df['NACCID'].unique())
        patient_ids = set(patient_df['NACCID'].unique())
        validations['all_visits_have_patient'] = visit_patients.issubset(patient_ids)
        
        # All visits should have a corresponding ADC
        visit_adcs = set(visit_df['NACCADC'].unique())
        adc_ids = set(adc_df['NACCADC'].unique())
        validations['all_visits_have_adc'] = visit_adcs.issubset(adc_ids)
        
        # Check data consistency
        # Visit counts should match
        patient_visit_counts = visit_df.groupby('NACCID').size()
        for idx, row in patient_df.iterrows():
            naccid = row['NACCID']
            if naccid in patient_visit_counts.index:
                actual_visits = patient_visit_counts[naccid]
                expected_visits = row.get('N_VISITS', 0)
                if actual_visits != expected_visits:
                    logger.warning(f"Visit count mismatch for patient {naccid}: "
                                 f"expected {expected_visits}, actual {actual_visits}")
                    
        # Log validation results
        for check, passed in validations.items():
            if passed:
                logger.info(f"✓ {check}")
            else:
                logger.error(f"✗ {check}")
                
        return validations
        
    def generate_quality_report(self, patient_df: pd.DataFrame, visit_df: pd.DataFrame, 
                              adc_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive data quality report.
        
        Args:
            patient_df: Patient-level DataFrame
            visit_df: Visit-level DataFrame
            adc_df: ADC-level DataFrame
            
        Returns:
            Dictionary with quality report DataFrames
        """
        logger.info("Generating data quality report...")
        
        report = {}
        
        # Missing data analysis
        report['patient_missing'] = self.check_missing_patterns(patient_df, 'patient')
        report['visit_missing'] = self.check_missing_patterns(visit_df, 'visit')
        report['adc_missing'] = self.check_missing_patterns(adc_df, 'adc')
        
        # Key variable completeness
        key_vars = {
            'patient': ['SEX', 'BIRTHYR', 'RACE', 'EDUC', 'NACCAPOE'],
            'visit': ['VISITYR', 'NACCVNUM']
        }
        
        completeness = []
        for table_name, vars in key_vars.items():
            df = patient_df if table_name == 'patient' else visit_df
            for var in vars:
                if var in df.columns:
                    completeness.append({
                        'table': table_name,
                        'variable': var,
                        'completeness': (df[var].notna().sum() / len(df)) * 100
                    })
                    
        report['key_var_completeness'] = pd.DataFrame(completeness)
        
        # MRI data availability
        if any(col.startswith('MRI_') for col in visit_df.columns):
            mri_cols = [col for col in visit_df.columns if col.startswith('MRI_')]
            mri_availability = []
            
            for col in mri_cols[:10]:  # Sample first 10 MRI columns
                mri_availability.append({
                    'mri_variable': col,
                    'n_available': visit_df[col].notna().sum(),
                    'percentage': (visit_df[col].notna().sum() / len(visit_df)) * 100
                })
                
            report['mri_availability'] = pd.DataFrame(mri_availability)
            
        return report
        
    def handle_missing_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert NACC missing codes to appropriate null values while preserving codes.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with handled missing codes
        """
        logger.info("Handling NACC missing codes...")
        
        # Create copy to avoid modifying original
        df_processed = df.copy()
        
        # For each numeric column, create a companion column to store missing codes
        for col in df_processed.columns:
            if df_processed[col].dtype in ['int64', 'float64']:
                # Check if column contains any missing codes
                has_missing_codes = any(
                    (df_processed[col] == code).any() 
                    for code in self.missing_codes.keys()
                )
                
                if has_missing_codes:
                    # Create missing code indicator column
                    code_col = f'{col}_MISSING_CODE'
                    df_processed[code_col] = np.nan
                    
                    # Store missing codes and replace with NaN
                    for code in self.missing_codes.keys():
                        mask = df_processed[col] == code
                        if mask.any():
                            df_processed.loc[mask, code_col] = code
                            df_processed.loc[mask, col] = np.nan
                            
        return df_processed


def validate_nacc_data(patient_df: pd.DataFrame, visit_df: pd.DataFrame, 
                      adc_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to run all data quality checks.
    
    Args:
        patient_df: Patient-level DataFrame
        visit_df: Visit-level DataFrame
        adc_df: ADC-level DataFrame
        
    Returns:
        Dictionary with quality report DataFrames
    """
    checker = DataQualityChecker()
    
    # Validate relationships
    validations = checker.validate_relationships(patient_df, visit_df, adc_df)
    
    # Generate quality report
    report = checker.generate_quality_report(patient_df, visit_df, adc_df)
    
    # Add validation results to report
    report['validations'] = pd.DataFrame([validations]).T.reset_index()
    report['validations'].columns = ['check', 'passed']
    
    return report