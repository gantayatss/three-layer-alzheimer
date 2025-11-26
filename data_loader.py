"""
data_loader.py - Data loading and preprocessing for NACC normalized tables
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class NACCDataLoader:
    """
    Loads and preprocesses NACC normalized tables for TMV-HNN model.
    """
    
    def __init__(self, patient_path: str, visit_path: str, adc_path: str = None):
        """
        Initialize data loader.
        
        Args:
            patient_path: Path to patient table CSV
            visit_path: Path to visit table CSV
            adc_path: Optional path to ADC table CSV
        """
        self.patient_path = patient_path
        self.visit_path = visit_path
        self.adc_path = adc_path
        
        # Define modality column groups
        self.demo_cols = ['SEX', 'BIRTHYR', 'RACE', 'HISPANIC', 'EDUC', 'HANDED']
        self.genetic_cols = ['NACCAPOE', 'NACCNE4S']
        self.cognitive_cols_prefixes = ['MMSE', 'MOCA', 'LOGIMEM', 'DIGIF', 'DIGIB', 
                                       'ANIMALS', 'VEG', 'TRAILA', 'TRAILB', 'BOSTON']
        self.symptom_cols_prefixes = ['DELUSION', 'HALLUCIN', 'AGIT', 'DEPD', 'ANX', 
                                      'ELAT', 'APATHY', 'DISINHIB', 'IRRITABL', 'MOTOR']
        self.functional_cols_prefixes = ['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 
                                        'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL']
        self.mri_col_prefix = 'MRI_'
        
        self.diagnosis_cols = ['NORMCOG', 'DEMENTED', 'NACCMCI', 'NACCALZD']
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all NACC tables.
        
        Returns:
            Dictionary with 'patient', 'visit', 'adc' DataFrames
        """
        logger.info("Loading NACC data...")
        
        patient_df = pd.read_csv(self.patient_path)
        visit_df = pd.read_csv(self.visit_path)
        
        logger.info(f"Loaded patient table: {len(patient_df)} patients")
        logger.info(f"Loaded visit table: {len(visit_df)} visits")
        
        data = {
            'patient': patient_df,
            'visit': visit_df
        }
        
        if self.adc_path:
            adc_df = pd.read_csv(self.adc_path)
            data['adc'] = adc_df
            logger.info(f"Loaded ADC table: {len(adc_df)} centers")
            
        return data
        
    def extract_modalities(self, patient_df: pd.DataFrame, visit_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Extract multi-modal features for each patient.
        
        Args:
            patient_df: Patient table
            visit_df: Visit table
            
        Returns:
            Dictionary mapping patient IDs to modality features
        """
        logger.info("Extracting multi-modal features...")
        
        patient_data = {}
        
        # Get column lists once
        cog_cols = [col for col in visit_df.columns 
                   if any(prefix in col for prefix in self.cognitive_cols_prefixes)]
        logger.info(f"Found {len(cog_cols)} cognitive test columns")
        
        mri_cols = [col for col in visit_df.columns if col.startswith(self.mri_col_prefix)]
        # Exclude non-numeric MRI columns
        exclude_patterns = ['DATE', 'DESCRIPTION', 'LONI_IMAGE', 'VERSION', 'NACCADC']
        mri_cols = [col for col in mri_cols 
                   if not any(pattern in col.upper() for pattern in exclude_patterns)]
        logger.info(f"Found {len(mri_cols)} numeric MRI columns")
        
        symptom_cols = [col for col in visit_df.columns 
                       if any(prefix in col for prefix in self.symptom_cols_prefixes) 
                       and not col.endswith('SEV')]
        logger.info(f"Found {len(symptom_cols)} symptom columns")
        
        func_cols = [col for col in visit_df.columns 
                    if any(prefix in col for prefix in self.functional_cols_prefixes)]
        logger.info(f"Found {len(func_cols)} functional status columns")
        
        processed_count = 0
        for naccid in patient_df['NACCID'].unique():
            patient_row = patient_df[patient_df['NACCID'] == naccid].iloc[0]
            patient_visits = visit_df[visit_df['NACCID'] == naccid].sort_values('NACCVNUM')
            
            # Extract baseline (first visit)
            if len(patient_visits) == 0:
                continue
                
            baseline_visit = patient_visits.iloc[0]
            
            # Demographics
            demo_features = self._extract_features(patient_row, self.demo_cols)
            
            # Genetics
            genetic_features = self._extract_features(patient_row, self.genetic_cols)
            
            # Baseline cognitive scores (use pre-computed cog_cols)
            cog_features = self._extract_features(baseline_visit, cog_cols)
            
            # Baseline MRI (use pre-computed mri_cols)
            mri_features = self._extract_features(baseline_visit, mri_cols)
            
            # Temporal sequences for all visits
            symptoms_sequence = []
            cognitive_sequence = []
            functional_sequence = []
            diagnosis_sequence = []
            time_points = []
            
            for idx, visit in patient_visits.iterrows():
                # Extract time point (year + fraction for month)
                time = visit['VISITYR'] + (visit['VISITMO'] - 1) / 12.0 if 'VISITMO' in visit and pd.notna(visit['VISITMO']) else visit['VISITYR']
                time_points.append(time)
                
                # Symptoms (binary) - use pre-computed symptom_cols
                symptoms = self._extract_features(visit, symptom_cols)
                symptoms_sequence.append(symptoms)
                
                # Cognitive scores - use pre-computed cog_cols
                cog = self._extract_features(visit, cog_cols)
                cognitive_sequence.append(cog)
                
                # Functional status - use pre-computed func_cols
                func = self._extract_features(visit, func_cols)
                functional_sequence.append(func)
                
                # Diagnosis
                diag = self._extract_features(visit, self.diagnosis_cols)
                diagnosis_sequence.append(diag)
            
            patient_data[naccid] = {
                'demographics': demo_features,
                'genetics': genetic_features,
                'baseline_cognition': cog_features,
                'baseline_mri': mri_features,
                'symptoms_sequence': symptoms_sequence,
                'cognitive_sequence': cognitive_sequence,
                'functional_sequence': functional_sequence,
                'diagnosis_sequence': diagnosis_sequence,
                'time_points': time_points,
                'n_visits': len(patient_visits)
            }
            
            processed_count += 1
            if processed_count % 1000 == 0:
                logger.info(f"Processed {processed_count} patients...")
            processed_count += 1
            if processed_count % 1000 == 0:
                logger.info(f"Processed {processed_count} patients...")
            
        logger.info(f"Extracted features for {len(patient_data)} patients")
        return patient_data
        
    def _extract_features(self, row: pd.Series, columns: List[str]) -> np.ndarray:
        """
        Extract feature vector from row, handling missing values.
        
        Args:
            row: Pandas Series
            columns: List of column names
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        for col in columns:
            if col in row.index:
                val = row[col]
                
                # Replace strings with NaN instead of skipping to maintain array length
                if isinstance(val, str):
                    features.append(np.nan)
                    continue
                    
                # Replace NACC missing codes with NaN
                if val in [-4, 88, 95, 96, 97, 98, 99]:
                    val = np.nan
                features.append(val)
            else:
                features.append(np.nan)
        
        # Convert to array - should always have len(columns) elements
        return np.array(features, dtype=float)
        
    def create_baseline_dataset(self, patient_data: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create baseline feature matrix for Layer 3.
        
        Args:
            patient_data: Dictionary from extract_modalities
            
        Returns:
            Tuple of (features, available_modality_mask, patient_ids)
        """
        logger.info("Creating baseline dataset for Layer 3...")
        
        patient_ids = list(patient_data.keys())
        all_features = []
        modality_masks = []
        
        for naccid in patient_ids:
            data = patient_data[naccid]
            
            # Concatenate all baseline modalities
            demo = data['demographics']
            gene = data['genetics']
            cog = data['baseline_cognition']
            mri = data['baseline_mri']
            
            # Create feature vector
            features = np.concatenate([demo, gene, cog, mri])
            all_features.append(features)
            
            # Track which modalities are available (not all NaN)
            mask = [
                not np.all(np.isnan(demo)),
                not np.all(np.isnan(gene)),
                not np.all(np.isnan(cog)),
                not np.all(np.isnan(mri))
            ]
            modality_masks.append(mask)
            
        feature_matrix = np.array(all_features)
        modality_mask = np.array(modality_masks, dtype=bool)
        
        # Impute missing values with median
        imputer = SimpleImputer(strategy='median')
        feature_matrix = imputer.fit_transform(feature_matrix)
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)
        
        logger.info(f"Created baseline dataset: {feature_matrix.shape}")
        return feature_matrix, modality_mask, patient_ids
        
    def create_sequence_dataset(self, patient_data: Dict) -> Dict:
        """
        Create temporal sequence data for Layer 2.
        
        Args:
            patient_data: Dictionary from extract_modalities
            
        Returns:
            Dictionary with sequence data
        """
        logger.info("Creating sequence dataset for Layer 2...")
        
        sequences = {}
        
        for naccid, data in patient_data.items():
            n_visits = data['n_visits']
            
            if n_visits < 2:
                continue  # Need at least 2 visits for temporal modeling
                
            # Combine symptoms and cognitive scores at each visit
            temporal_features = []
            for i in range(n_visits):
                symptoms = data['symptoms_sequence'][i]
                cognition = data['cognitive_sequence'][i]
                functional = data['functional_sequence'][i]
                
                # Concatenate
                visit_features = np.concatenate([symptoms, cognition, functional])
                temporal_features.append(visit_features)
                
            # Impute and standardize
            temporal_features = np.array(temporal_features)
            
            # Handle NaNs in sequences
            for j in range(temporal_features.shape[1]):
                col = temporal_features[:, j]
                if np.all(np.isnan(col)):
                    temporal_features[:, j] = 0
                else:
                    mask = ~np.isnan(col)
                    if mask.sum() > 0:
                        median_val = np.median(col[mask])
                        temporal_features[~mask, j] = median_val
                        
            sequences[naccid] = {
                'features': temporal_features,
                'time_points': np.array(data['time_points']),
                'diagnoses': data['diagnosis_sequence']
            }
            
        logger.info(f"Created sequences for {len(sequences)} patients")
        return sequences
        
    def create_survival_data(self, patient_df: pd.DataFrame, visit_df: pd.DataFrame) -> Dict:
        """
        Create survival analysis data.
        
        Args:
            patient_df: Patient table
            visit_df: Visit table
            
        Returns:
            Dictionary with survival times and events
        """
        logger.info("Creating survival data...")
        
        survival_data = {}
        
        for naccid in patient_df['NACCID'].unique():
            patient_row = patient_df[patient_df['NACCID'] == naccid].iloc[0]
            patient_visits = visit_df[visit_df['NACCID'] == naccid].sort_values('NACCVNUM')
            
            if len(patient_visits) == 0:
                continue
                
            # Determine if patient progressed to MCI or AD
            diagnoses = []
            for _, visit in patient_visits.iterrows():
                if 'NACCALZD' in visit and visit['NACCALZD'] == 1:
                    diagnoses.append('AD')
                elif 'NACCMCI' in visit and visit['NACCMCI'] == 1:
                    diagnoses.append('MCI')
                elif 'NORMCOG' in visit and visit['NORMCOG'] == 1:
                    diagnoses.append('Normal')
                else:
                    diagnoses.append('Unknown')
                    
            # Time to event (first MCI or AD diagnosis)
            baseline_year = patient_visits.iloc[0]['VISITYR']
            event_occurred = False
            event_time = None
            
            for idx, diag in enumerate(diagnoses):
                if diag in ['MCI', 'AD']:
                    event_occurred = True
                    event_time = patient_visits.iloc[idx]['VISITYR'] - baseline_year
                    break
                    
            # If no event, use last visit time (censored)
            if not event_occurred:
                event_time = patient_visits.iloc[-1]['VISITYR'] - baseline_year
                
            survival_data[naccid] = {
                'time': max(event_time, 0.1),  # Ensure positive time
                'event': 1 if event_occurred else 0,
                'diagnoses': diagnoses
            }
            
        logger.info(f"Created survival data for {len(survival_data)} patients")
        return survival_data