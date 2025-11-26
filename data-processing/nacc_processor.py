
# nacc_processor.py
"""
NACC Dataset Processor - FIXED VERSION
Main module for implementing the dataset preparation described in the LaTeX document.
Creates normalized three-table structure from NACC UDS and SCAN MRI data.

FIXES:
- Eliminated DataFrame fragmentation warnings
- Fixed dtype compatibility issues
- Optimized MRI column creation using pd.concat
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class NACCDataProcessor:
    """
    Processes NACC UDS and SCAN MRI datasets to create normalized three-table structure.
    """
    
    def __init__(self, uds_path: str, mri_path: Optional[str] = None, temporal_threshold: int = 90):
        """
        Initialize the NACC data processor.
        
        Args:
            uds_path: Path to UDS clinical dataset
            mri_path: Optional path to SCAN MRI dataset
            temporal_threshold: Days threshold for matching MRI to visits (default: 90)
        """
        self.uds_path = uds_path
        self.mri_path = mri_path
        self.temporal_threshold = temporal_threshold
        
        # Define attribute sets for patient-level extraction
        self.static_attributes = [
            'NACCID', 'SEX', 'BIRTHMO', 'BIRTHYR', 
            'RACE', 'RACESEC', 'RACETER', 'HISPANIC',
            'HISPOR', 'HANDED'
        ]
        
        self.genetic_attributes = [
            'NACCAPOE', 'NACCNE4S', 'NGENEFAM',
            'NGENEFMO', 'NGENECOU', 'NGENESIB', 'NGENEKID'
        ]
        
        self.death_attributes = [
            'NACCDIED', 'NACCMODY', 'NACCMOD', 'NACCMOY'
        ]
        
        self.biomarker_flags = [
            'NACCACSF', 'NACCPCSF', 'NACCTCSF',
            'NACCMRSA', 'NACCNMRI', 'NACCAPSA', 'NACCNAPA'
        ]
        
        self.special_modules = [
            'NACCFTD', 'NACCLBDM', 'NACCAUTP'
        ]
        
        self.visit_identifiers = [
            'NACCID', 'NACCADC', 'NACCVNUM', 'VISITMO',
            'VISITDAY', 'VISITYR', 'PACKET'
        ]
        
    def load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load UDS and MRI datasets.
        
        Returns:
            Tuple of (UDS DataFrame, MRI DataFrame or None)
        """
        logger.info(f"Loading UDS data from {self.uds_path}")
        try:
            # Load with low_memory=False to avoid dtype warnings
            uds_df = pd.read_csv(self.uds_path, low_memory=False)
            logger.info(f"Loaded UDS data: {len(uds_df)} visits, {uds_df['NACCID'].nunique()} unique participants")
        except Exception as e:
            logger.error(f"Error loading UDS data: {e}")
            raise
            
        mri_df = None
        if self.mri_path:
            logger.info(f"Loading MRI data from {self.mri_path}")
            try:
                mri_df = pd.read_csv(self.mri_path, low_memory=False)
                logger.info(f"Loaded MRI data: {len(mri_df)} scans, {mri_df['NACCID'].nunique()} unique participants")
            except Exception as e:
                logger.warning(f"Error loading MRI data: {e}")
                
        return uds_df, mri_df
        
    def extract_patient_level_data(self, uds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Algorithm 1: Extract patient-level data from UDS dataset.
        
        Args:
            uds_df: UDS clinical dataset
            
        Returns:
            Patient-level DataFrame with one row per unique participant
        """
        logger.info("Extracting patient-level data...")
        
        # Combine all patient-level attribute sets
        patient_attributes = (
            self.static_attributes + 
            self.genetic_attributes + 
            self.death_attributes + 
            self.biomarker_flags + 
            self.special_modules
        )
        
        # Initialize patient table
        patient_data = []
        
        # Group by patient ID
        for naccid, patient_visits in uds_df.groupby('NACCID'):
            # Sort visits by visit number descending to get most recent first
            patient_visits = patient_visits.sort_values('NACCVNUM', ascending=False)
            latest_visit = patient_visits.iloc[0]
            
            # Extract patient attributes from latest visit
            patient_row = {}
            for attr in patient_attributes:
                if attr in latest_visit:
                    patient_row[attr] = latest_visit[attr]
                    
            # Compute visit statistics
            patient_row['N_VISITS'] = len(patient_visits)
            patient_row['FIRST_VISITNUM'] = patient_visits['NACCVNUM'].min()
            patient_row['LAST_VISITNUM'] = patient_visits['NACCVNUM'].max()
            patient_row['FIRST_VISITYR'] = patient_visits['VISITYR'].min()
            patient_row['LAST_VISITYR'] = patient_visits['VISITYR'].max()
            
            # Determine primary ADC (most frequent)
            adc_counts = patient_visits['NACCADC'].value_counts()
            patient_row['PRIMARY_NACCADC'] = adc_counts.index[0]
            
            # Extract most recent non-missing education value
            educ_values = patient_visits['EDUC'].dropna() if 'EDUC' in patient_visits else pd.Series()
            if len(educ_values) > 0:
                patient_row['EDUC'] = educ_values.iloc[0]
                
            patient_data.append(patient_row)
            
        patient_df = pd.DataFrame(patient_data)
        logger.info(f"Created patient table: {len(patient_df)} patients, {len(patient_df.columns)} columns")
        
        return patient_df
        
    def match_mri_to_visits(self, visit_df: pd.DataFrame, mri_df: pd.DataFrame) -> pd.DataFrame:
        """
        Algorithm 2: Match MRI scans to clinical visits within temporal threshold.
        OPTIMIZED: Uses pd.concat to avoid DataFrame fragmentation
        
        Args:
            visit_df: Visit-level DataFrame
            mri_df: MRI dataset
            
        Returns:
            Visit DataFrame with integrated MRI data
        """
        logger.info(f"Matching MRI scans to visits with {self.temporal_threshold}-day threshold...")
        
        # Parse dates
        visit_df['VISIT_DATE'] = pd.to_datetime(
            visit_df[['VISITYR', 'VISITMO', 'VISITDAY']].rename(
                columns={'VISITYR': 'year', 'VISITMO': 'month', 'VISITDAY': 'day'}
            ),
            errors='coerce'
        )
        
        # Parse MRI scan dates - handle different date formats
        if 'SCANDT' in mri_df.columns:
            mri_df['SCAN_DATE'] = pd.to_datetime(mri_df['SCANDT'], errors='coerce')
        elif 'STUDYDATE' in mri_df.columns:
            mri_df['SCAN_DATE'] = pd.to_datetime(mri_df['STUDYDATE'], errors='coerce')
        else:
            logger.error("No scan date column found in MRI data")
            return visit_df
            
        # Prepare MRI columns with prefix
        mri_columns = [col for col in mri_df.columns if col not in ['NACCID', 'NACCADC', 'SCAN_DATE']]
        
        # Create dictionary to store MRI data for each visit
        mri_data_dict = {}
        
        # Match MRI scans to visits for each patient
        matched_count = 0
        
        for naccid in visit_df['NACCID'].unique():
            patient_visits = visit_df[visit_df['NACCID'] == naccid]
            patient_mris = mri_df[mri_df['NACCID'] == naccid]
            
            if len(patient_mris) == 0:
                continue
                
            for visit_idx in patient_visits.index:
                visit_date = visit_df.loc[visit_idx, 'VISIT_DATE']
                
                if pd.isna(visit_date):
                    continue
                
                # Find MRI scans within temporal window
                time_diffs = abs((patient_mris['SCAN_DATE'] - visit_date).dt.days)
                candidates = patient_mris[time_diffs <= self.temporal_threshold]
                
                if len(candidates) > 0:
                    # Select scan with minimum temporal difference
                    closest_idx = time_diffs[candidates.index].idxmin()
                    closest_mri = patient_mris.loc[closest_idx]
                    
                    # Store MRI data for this visit
                    mri_data_dict[visit_idx] = {}
                    for col in mri_columns:
                        if col in closest_mri:
                            # Convert to appropriate type to avoid dtype warnings
                            value = closest_mri[col]
                            if isinstance(value, (int, float, np.integer, np.floating)):
                                mri_data_dict[visit_idx][f'MRI_{col}'] = float(value) if pd.notna(value) else np.nan
                            else:
                                mri_data_dict[visit_idx][f'MRI_{col}'] = str(value) if pd.notna(value) else None
                    
                    mri_data_dict[visit_idx]['MRI_SCAN_DATE'] = closest_mri['SCAN_DATE']
                    matched_count += 1
        
        # OPTIMIZED: Create MRI DataFrame and merge at once (no fragmentation)
        if mri_data_dict:
            mri_columns_df = pd.DataFrame.from_dict(mri_data_dict, orient='index')
            
            # Ensure proper dtypes for numeric columns
            for col in mri_columns_df.columns:
                if col != 'MRI_SCAN_DATE' and not col.endswith('_DESCRIPTION') and not col.startswith('MRI_LONI'):
                    mri_columns_df[col] = pd.to_numeric(mri_columns_df[col], errors='coerce')
            
            # Merge with visit_df (this is much more efficient than repeated assignments)
            visit_df = visit_df.join(mri_columns_df, how='left')
        
        logger.info(f"Matched {matched_count} MRI scans to visits ({matched_count/len(visit_df)*100:.1f}%)")
        
        # Drop temporary date column
        visit_df = visit_df.drop(columns=['VISIT_DATE'])
        
        return visit_df
        
    def extract_visit_level_data(self, uds_df: pd.DataFrame, mri_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract visit-level data and integrate MRI measurements if available.
        
        Args:
            uds_df: UDS clinical dataset
            mri_df: Optional MRI dataset
            
        Returns:
            Visit-level DataFrame with clinical and MRI data
        """
        logger.info("Extracting visit-level data...")
        
        # Get all columns except patient-level attributes
        patient_attributes = (
            self.static_attributes + 
            self.genetic_attributes + 
            self.death_attributes + 
            self.biomarker_flags + 
            self.special_modules
        )
        
        # Keep visit identifiers and all clinical data not in patient table
        visit_columns = [col for col in uds_df.columns 
                        if col in self.visit_identifiers or col not in patient_attributes]
        
        visit_df = uds_df[visit_columns].copy()
        
        # Integrate MRI data if available
        if mri_df is not None:
            visit_df = self.match_mri_to_visits(visit_df, mri_df)
            
        logger.info(f"Created visit table: {len(visit_df)} visits, {len(visit_df.columns)} columns")
        
        return visit_df
        
    def extract_adc_level_data(self, uds_df: pd.DataFrame, mri_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Algorithm 3: Extract ADC center-level aggregation data.
        
        Args:
            uds_df: UDS clinical dataset
            mri_df: Optional MRI dataset
            
        Returns:
            ADC-level DataFrame with center statistics
        """
        logger.info("Extracting ADC-level data...")
        
        adc_data = []
        
        for adc in uds_df['NACCADC'].unique():
            adc_visits = uds_df[uds_df['NACCADC'] == adc]
            
            adc_row = {
                'NACCADC': adc,
                'N_VISITS': len(adc_visits),
                'N_PATIENTS': adc_visits['NACCID'].nunique(),
                'FIRST_VISITYR': adc_visits['VISITYR'].min(),
                'LAST_VISITYR': adc_visits['VISITYR'].max()
            }
            
            # Count MRI scans if available
            if mri_df is not None and 'NACCADC' in mri_df.columns:
                adc_mris = mri_df[mri_df['NACCADC'] == adc]
                adc_row['N_MRI_SCANS'] = len(adc_mris)
            else:
                adc_row['N_MRI_SCANS'] = 0
                
            adc_data.append(adc_row)
            
        adc_df = pd.DataFrame(adc_data)
        logger.info(f"Created ADC table: {len(adc_df)} centers, {len(adc_df.columns)} columns")
        
        return adc_df
        
    def process_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Main processing function to create normalized three-table structure.
        
        Returns:
            Dictionary with 'patient', 'visit', and 'adc' DataFrames
        """
        logger.info("Starting NACC dataset processing...")
        
        # Load data
        uds_df, mri_df = self.load_data()
        
        # Extract three normalized tables
        patient_df = self.extract_patient_level_data(uds_df)
        visit_df = self.extract_visit_level_data(uds_df, mri_df)
        adc_df = self.extract_adc_level_data(uds_df, mri_df)
        
        # Print summary statistics
        logger.info("\nProcessing complete. Summary statistics:")
        logger.info(f"Patient table: {len(patient_df)} rows, {len(patient_df.columns)} columns")
        logger.info(f"Visit table: {len(visit_df)} rows, {len(visit_df.columns)} columns")
        logger.info(f"ADC table: {len(adc_df)} rows, {len(adc_df.columns)} columns")
        
        # Check MRI integration
        if mri_df is not None:
            mri_cols = [col for col in visit_df.columns if col.startswith('MRI_')]
            if mri_cols:
                visits_with_mri = visit_df[mri_cols[0]].notna().sum()
                logger.info(f"Visits with MRI data: {visits_with_mri} ({visits_with_mri/len(visit_df)*100:.1f}%)")
        
        return {
            'patient': patient_df,
            'visit': visit_df,
            'adc': adc_df
        }

# # nacc_processor.py
# """
# NACC Dataset Processor
# Main module for implementing the dataset preparation described in the LaTeX document.
# Creates normalized three-table structure from NACC UDS and SCAN MRI data.
# """

# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import logging
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


# class NACCDataProcessor:
#     """
#     Processes NACC UDS and SCAN MRI datasets to create normalized three-table structure.
#     """
    
#     def __init__(self, uds_path: str, mri_path: Optional[str] = None, temporal_threshold: int = 90):
#         """
#         Initialize the NACC data processor.
        
#         Args:
#             uds_path: Path to UDS clinical dataset
#             mri_path: Optional path to SCAN MRI dataset
#             temporal_threshold: Days threshold for matching MRI to visits (default: 90)
#         """
#         self.uds_path = uds_path
#         self.mri_path = mri_path
#         self.temporal_threshold = temporal_threshold
        
#         # Define attribute sets for patient-level extraction
#         self.static_attributes = [
#             'NACCID', 'SEX', 'BIRTHMO', 'BIRTHYR', 
#             'RACE', 'RACESEC', 'RACETER', 'HISPANIC',
#             'HISPOR', 'HANDED'
#         ]
        
#         self.genetic_attributes = [
#             'NACCAPOE', 'NACCNE4S', 'NGENEFAM',
#             'NGENEFMO', 'NGENECOU', 'NGENESIB', 'NGENEKID'
#         ]
        
#         self.death_attributes = [
#             'NACCDIED', 'NACCMODY', 'NACCMOD', 'NACCMOY'
#         ]
        
#         self.biomarker_flags = [
#             'NACCACSF', 'NACCPCSF', 'NACCTCSF',
#             'NACCMRSA', 'NACCNMRI', 'NACCAPSA', 'NACCNAPA'
#         ]
        
#         self.special_modules = [
#             'NACCFTD', 'NACCLBDM', 'NACCAUTP'
#         ]
        
#         self.visit_identifiers = [
#             'NACCID', 'NACCADC', 'NACCVNUM', 'VISITMO',
#             'VISITDAY', 'VISITYR', 'PACKET'
#         ]
        
#     def load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
#         """
#         Load UDS and MRI datasets.
        
#         Returns:
#             Tuple of (UDS DataFrame, MRI DataFrame or None)
#         """
#         logger.info(f"Loading UDS data from {self.uds_path}")
#         try:
#             uds_df = pd.read_csv(self.uds_path)
#             logger.info(f"Loaded UDS data: {len(uds_df)} visits, {uds_df['NACCID'].nunique()} unique participants")
#         except Exception as e:
#             logger.error(f"Error loading UDS data: {e}")
#             raise
            
#         mri_df = None
#         if self.mri_path:
#             logger.info(f"Loading MRI data from {self.mri_path}")
#             try:
#                 mri_df = pd.read_csv(self.mri_path)
#                 logger.info(f"Loaded MRI data: {len(mri_df)} scans, {mri_df['NACCID'].nunique()} unique participants")
#             except Exception as e:
#                 logger.warning(f"Error loading MRI data: {e}")
                
#         return uds_df, mri_df
        
#     def extract_patient_level_data(self, uds_df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Algorithm 1: Extract patient-level data from UDS dataset.
        
#         Args:
#             uds_df: UDS clinical dataset
            
#         Returns:
#             Patient-level DataFrame with one row per unique participant
#         """
#         logger.info("Extracting patient-level data...")
        
#         # Combine all patient-level attribute sets
#         patient_attributes = (
#             self.static_attributes + 
#             self.genetic_attributes + 
#             self.death_attributes + 
#             self.biomarker_flags + 
#             self.special_modules
#         )
        
#         # Initialize patient table
#         patient_data = []
        
#         # Group by patient ID
#         for naccid, patient_visits in uds_df.groupby('NACCID'):
#             # Sort visits by visit number descending to get most recent first
#             patient_visits = patient_visits.sort_values('NACCVNUM', ascending=False)
#             latest_visit = patient_visits.iloc[0]
            
#             # Extract patient attributes from latest visit
#             patient_row = {}
#             for attr in patient_attributes:
#                 if attr in latest_visit:
#                     patient_row[attr] = latest_visit[attr]
                    
#             # Compute visit statistics
#             patient_row['N_VISITS'] = len(patient_visits)
#             patient_row['FIRST_VISITNUM'] = patient_visits['NACCVNUM'].min()
#             patient_row['LAST_VISITNUM'] = patient_visits['NACCVNUM'].max()
#             patient_row['FIRST_VISITYR'] = patient_visits['VISITYR'].min()
#             patient_row['LAST_VISITYR'] = patient_visits['VISITYR'].max()
            
#             # Determine primary ADC (most frequent)
#             adc_counts = patient_visits['NACCADC'].value_counts()
#             patient_row['PRIMARY_NACCADC'] = adc_counts.index[0]
            
#             # Extract most recent non-missing education value
#             educ_values = patient_visits['EDUC'].dropna() if 'EDUC' in patient_visits else pd.Series()
#             if len(educ_values) > 0:
#                 patient_row['EDUC'] = educ_values.iloc[0]
                
#             patient_data.append(patient_row)
            
#         patient_df = pd.DataFrame(patient_data)
#         logger.info(f"Created patient table: {len(patient_df)} patients, {len(patient_df.columns)} columns")
        
#         return patient_df
        
#     def match_mri_to_visits(self, visit_df: pd.DataFrame, mri_df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Algorithm 2: Match MRI scans to clinical visits within temporal threshold.
        
#         Args:
#             visit_df: Visit-level DataFrame
#             mri_df: MRI dataset
            
#         Returns:
#             Visit DataFrame with integrated MRI data
#         """
#         logger.info(f"Matching MRI scans to visits with {self.temporal_threshold}-day threshold...")
        
#         # Parse dates
#         visit_df['VISIT_DATE'] = pd.to_datetime(
#             visit_df[['VISITYR', 'VISITMO', 'VISITDAY']].rename(
#                 columns={'VISITYR': 'year', 'VISITMO': 'month', 'VISITDAY': 'day'}
#             )
#         )
        
#         # Parse MRI scan dates - handle different date formats
#         if 'SCANDT' in mri_df.columns:
#             mri_df['SCAN_DATE'] = pd.to_datetime(mri_df['SCANDT'])
#         elif 'STUDYDATE' in mri_df.columns:
#             mri_df['SCAN_DATE'] = pd.to_datetime(mri_df['STUDYDATE'])
#         else:
#             logger.error("No scan date column found in MRI data")
#             return visit_df
            
#         # Prepare MRI columns with prefix
#         mri_columns = [col for col in mri_df.columns if col not in ['NACCID', 'NACCADC']]
#         mri_rename_dict = {col: f'MRI_{col}' for col in mri_columns if not col.startswith('MRI_')}
        
#         # Special handling for scan date
#         mri_rename_dict['SCAN_DATE'] = 'MRI_SCAN_DATE'
        
#         # Initialize MRI columns in visit_df
#         for new_col in mri_rename_dict.values():
#             visit_df[new_col] = np.nan
            
#         # Match MRI scans to visits for each patient
#         matched_count = 0
#         for naccid in visit_df['NACCID'].unique():
#             patient_visits = visit_df[visit_df['NACCID'] == naccid]
#             patient_mris = mri_df[mri_df['NACCID'] == naccid]
            
#             if len(patient_mris) == 0:
#                 continue
                
#             for visit_idx in patient_visits.index:
#                 visit_date = visit_df.loc[visit_idx, 'VISIT_DATE']
                
#                 # Find MRI scans within temporal window
#                 time_diffs = abs((patient_mris['SCAN_DATE'] - visit_date).dt.days)
#                 candidates = patient_mris[time_diffs <= self.temporal_threshold]
                
#                 if len(candidates) > 0:
#                     # Select scan with minimum temporal difference
#                     closest_idx = time_diffs[candidates.index].idxmin()
#                     closest_mri = patient_mris.loc[closest_idx]
                    
#                     # Merge MRI data into visit row
#                     for orig_col, new_col in mri_rename_dict.items():
#                         if orig_col in closest_mri:
#                             visit_df.loc[visit_idx, new_col] = closest_mri[orig_col]
#                     matched_count += 1
                    
#         logger.info(f"Matched {matched_count} MRI scans to visits ({matched_count/len(visit_df)*100:.1f}%)")
        
#         # Drop temporary date column
#         visit_df = visit_df.drop(columns=['VISIT_DATE'])
        
#         return visit_df
        
#     def extract_visit_level_data(self, uds_df: pd.DataFrame, mri_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
#         """
#         Extract visit-level data and integrate MRI measurements if available.
        
#         Args:
#             uds_df: UDS clinical dataset
#             mri_df: Optional MRI dataset
            
#         Returns:
#             Visit-level DataFrame with clinical and MRI data
#         """
#         logger.info("Extracting visit-level data...")
        
#         # Get all columns except patient-level attributes
#         patient_attributes = (
#             self.static_attributes + 
#             self.genetic_attributes + 
#             self.death_attributes + 
#             self.biomarker_flags + 
#             self.special_modules
#         )
        
#         # Keep visit identifiers and all clinical data not in patient table
#         visit_columns = [col for col in uds_df.columns 
#                         if col in self.visit_identifiers or col not in patient_attributes]
        
#         visit_df = uds_df[visit_columns].copy()
        
#         # Integrate MRI data if available
#         if mri_df is not None:
#             visit_df = self.match_mri_to_visits(visit_df, mri_df)
            
#         logger.info(f"Created visit table: {len(visit_df)} visits, {len(visit_df.columns)} columns")
        
#         return visit_df
        
#     def extract_adc_level_data(self, uds_df: pd.DataFrame, mri_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
#         """
#         Algorithm 3: Extract ADC center-level aggregation data.
        
#         Args:
#             uds_df: UDS clinical dataset
#             mri_df: Optional MRI dataset
            
#         Returns:
#             ADC-level DataFrame with center statistics
#         """
#         logger.info("Extracting ADC-level data...")
        
#         adc_data = []
        
#         for adc in uds_df['NACCADC'].unique():
#             adc_visits = uds_df[uds_df['NACCADC'] == adc]
            
#             adc_row = {
#                 'NACCADC': adc,
#                 'N_VISITS': len(adc_visits),
#                 'N_PATIENTS': adc_visits['NACCID'].nunique(),
#                 'FIRST_VISITYR': adc_visits['VISITYR'].min(),
#                 'LAST_VISITYR': adc_visits['VISITYR'].max()
#             }
            
#             # Count MRI scans if available
#             if mri_df is not None and 'NACCADC' in mri_df.columns:
#                 adc_mris = mri_df[mri_df['NACCADC'] == adc]
#                 adc_row['N_MRI_SCANS'] = len(adc_mris)
#             else:
#                 adc_row['N_MRI_SCANS'] = 0
                
#             adc_data.append(adc_row)
            
#         adc_df = pd.DataFrame(adc_data)
#         logger.info(f"Created ADC table: {len(adc_df)} centers, {len(adc_df.columns)} columns")
        
#         return adc_df
        
#     def process_datasets(self) -> Dict[str, pd.DataFrame]:
#         """
#         Main processing function to create normalized three-table structure.
        
#         Returns:
#             Dictionary with 'patient', 'visit', and 'adc' DataFrames
#         """
#         logger.info("Starting NACC dataset processing...")
        
#         # Load data
#         uds_df, mri_df = self.load_data()
        
#         # Extract three normalized tables
#         patient_df = self.extract_patient_level_data(uds_df)
#         visit_df = self.extract_visit_level_data(uds_df, mri_df)
#         adc_df = self.extract_adc_level_data(uds_df, mri_df)
        
#         # Print summary statistics
#         logger.info("\nProcessing complete. Summary statistics:")
#         logger.info(f"Patient table: {len(patient_df)} rows, {len(patient_df.columns)} columns")
#         logger.info(f"Visit table: {len(visit_df)} rows, {len(visit_df.columns)} columns")
#         logger.info(f"ADC table: {len(adc_df)} rows, {len(adc_df.columns)} columns")
        
#         # Check MRI integration
#         if mri_df is not None:
#             mri_cols = [col for col in visit_df.columns if col.startswith('MRI_')]
#             visits_with_mri = visit_df[mri_cols[0]].notna().sum() if mri_cols else 0
#             logger.info(f"Visits with MRI data: {visits_with_mri} ({visits_with_mri/len(visit_df)*100:.1f}%)")
        
#         return {
#             'patient': patient_df,
#             'visit': visit_df,
#             'adc': adc_df
#         }