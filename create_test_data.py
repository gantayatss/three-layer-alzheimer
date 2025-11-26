#!/usr/bin/env python3
"""
create_test_data.py - Generate synthetic NACC data for testing TMV-HNN

This creates small sample CSV files that match the NACC data structure,
allowing you to test the TMV-HNN code without real NACC data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def create_sample_patient_table(n_patients=100):
    """Create sample patient table."""
    print(f"Generating {n_patients} sample patients...")
    
    np.random.seed(42)
    
    data = {
        'NACCID': [f'TEST{i:06d}' for i in range(n_patients)],
        'SEX': np.random.choice([1, 2], n_patients),  # 1=Male, 2=Female
        'BIRTHYR': np.random.randint(1930, 1960, n_patients),
        'BIRTHMO': np.random.randint(1, 13, n_patients),
        'RACE': np.random.choice([1, 2, 3, 4], n_patients),  # Race codes
        'HISPANIC': np.random.choice([0, 1], n_patients, p=[0.85, 0.15]),
        'EDUC': np.random.randint(8, 21, n_patients),  # Years of education
        'HANDED': np.random.choice([1, 2], n_patients, p=[0.1, 0.9]),  # 1=Left, 2=Right
        'NACCAPOE': np.random.choice([22, 23, 24, 33, 34, 44, 99], n_patients, 
                                     p=[0.05, 0.25, 0.15, 0.3, 0.15, 0.05, 0.05]),
        'NACCNE4S': np.random.choice([0, 1, 2, 9], n_patients, p=[0.5, 0.3, 0.1, 0.1]),
        'NACCDIED': np.random.choice([0, 1], n_patients, p=[0.85, 0.15]),
        'N_VISITS': np.random.randint(1, 15, n_patients),
        'FIRST_VISITYR': np.random.randint(2005, 2015, n_patients),
        'LAST_VISITYR': np.random.randint(2015, 2025, n_patients),
        'PRIMARY_NACCADC': np.random.choice(range(1, 11), n_patients)
    }
    
    df = pd.DataFrame(data)
    return df


def create_sample_visit_table(patient_df):
    """Create sample visit table."""
    print("Generating sample visits...")
    
    np.random.seed(42)
    visits = []
    
    for _, patient in patient_df.iterrows():
        naccid = patient['NACCID']
        n_visits = patient['N_VISITS']
        first_year = patient['FIRST_VISITYR']
        
        for v in range(1, n_visits + 1):
            visit_year = first_year + v - 1
            
            # Create baseline cognitive decline
            decline_factor = (v - 1) * 0.1
            
            visit_data = {
                'NACCID': naccid,
                'NACCADC': patient['PRIMARY_NACCADC'],
                'NACCVNUM': v,
                'VISITYR': visit_year,
                'VISITMO': np.random.randint(1, 13),
                'VISITDAY': np.random.randint(1, 29),
                'PACKET': 'I' if v == 1 else 'F',
                
                # Cognitive tests (declining over time)
                'MMSE': max(0, int(28 - decline_factor * np.random.uniform(0, 5))),
                'MOCA': max(0, int(26 - decline_factor * np.random.uniform(0, 5))),
                'LOGIMEM': max(0, int(20 - decline_factor * np.random.uniform(0, 3))),
                'DIGIF': max(0, int(8 - decline_factor * np.random.uniform(0, 2))),
                'DIGIB': max(0, int(7 - decline_factor * np.random.uniform(0, 2))),
                'ANIMALS': max(0, int(18 - decline_factor * np.random.uniform(0, 4))),
                'VEG': max(0, int(15 - decline_factor * np.random.uniform(0, 3))),
                'TRAILA': min(300, int(35 + decline_factor * np.random.uniform(0, 20))),
                'TRAILB': min(300, int(85 + decline_factor * np.random.uniform(0, 40))),
                'BOSTON': max(0, int(28 - decline_factor * np.random.uniform(0, 3))),
                
                # Symptoms (increasing over time)
                'DELUSION': 1 if np.random.random() < decline_factor * 0.3 else 0,
                'HALLUCIN': 1 if np.random.random() < decline_factor * 0.2 else 0,
                'AGIT': 1 if np.random.random() < decline_factor * 0.4 else 0,
                'DEPD': 1 if np.random.random() < decline_factor * 0.5 else 0,
                'ANX': 1 if np.random.random() < decline_factor * 0.4 else 0,
                'ELAT': 1 if np.random.random() < decline_factor * 0.1 else 0,
                'APATHY': 1 if np.random.random() < decline_factor * 0.5 else 0,
                'DISINHIB': 1 if np.random.random() < decline_factor * 0.3 else 0,
                'IRRITABL': 1 if np.random.random() < decline_factor * 0.4 else 0,
                'MOTOR': 1 if np.random.random() < decline_factor * 0.3 else 0,
                
                # Functional status (declining)
                'BILLS': min(3, int(decline_factor * np.random.uniform(0, 2))),
                'TAXES': min(3, int(decline_factor * np.random.uniform(0, 2))),
                'SHOPPING': min(3, int(decline_factor * np.random.uniform(0, 2))),
                'GAMES': min(3, int(decline_factor * np.random.uniform(0, 2))),
                'STOVE': min(3, int(decline_factor * np.random.uniform(0, 2))),
                'MEALPREP': min(3, int(decline_factor * np.random.uniform(0, 2))),
                'EVENTS': min(3, int(decline_factor * np.random.uniform(0, 2))),
                'PAYATTN': min(3, int(decline_factor * np.random.uniform(0, 2))),
                'REMDATES': min(3, int(decline_factor * np.random.uniform(0, 2))),
                'TRAVEL': min(3, int(decline_factor * np.random.uniform(0, 2))),
                
                # Diagnosis (progressing)
                'NORMCOG': 1 if decline_factor < 0.3 else 0,
                'NACCMCI': 1 if 0.3 <= decline_factor < 0.7 else 0,
                'NACCALZD': 1 if decline_factor >= 0.7 else 0,
                'DEMENTED': 1 if decline_factor >= 0.7 else 0,
            }
            
            # Add MRI data for some visits (20% of visits)
            if np.random.random() < 0.2:
                visit_data.update({
                    'MRI_HIPPOCAMPUS': np.random.uniform(6000, 7500),
                    'MRI_GM': np.random.uniform(500000, 600000),
                    'MRI_WMH': np.random.uniform(1000, 10000),
                    'MRI_CEREBRUMTCV': np.random.uniform(1000000, 1200000),
                })
            
            visits.append(visit_data)
    
    df = pd.DataFrame(visits)
    return df


def create_sample_adc_table(patient_df):
    """Create sample ADC table."""
    print("Generating sample ADC centers...")
    
    adc_ids = patient_df['PRIMARY_NACCADC'].unique()
    
    data = {
        'NACCADC': adc_ids,
        'N_VISITS': [patient_df[patient_df['PRIMARY_NACCADC'] == adc]['N_VISITS'].sum() 
                     for adc in adc_ids],
        'N_PATIENTS': [len(patient_df[patient_df['PRIMARY_NACCADC'] == adc]) 
                       for adc in adc_ids],
        'FIRST_VISITYR': [patient_df[patient_df['PRIMARY_NACCADC'] == adc]['FIRST_VISITYR'].min() 
                          for adc in adc_ids],
        'LAST_VISITYR': [patient_df[patient_df['PRIMARY_NACCADC'] == adc]['LAST_VISITYR'].max() 
                         for adc in adc_ids],
    }
    
    df = pd.DataFrame(data)
    return df


def main():
    """Main function to generate test data."""
    print("=" * 80)
    print("GENERATING SYNTHETIC NACC TEST DATA")
    print("=" * 80)
    print("\n⚠️  WARNING: This is synthetic data for testing only!")
    print("Do NOT use for actual research or publication.\n")
    
    # Configuration
    n_patients = 500  # Number of test patients
    output_dir = Path('test_data')
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}\n")
    
    # Generate data
    patient_df = create_sample_patient_table(n_patients)
    visit_df = create_sample_visit_table(patient_df)
    adc_df = create_sample_adc_table(patient_df)
    
    # Save to CSV
    patient_path = output_dir / 'test_patient_table.csv'
    visit_path = output_dir / 'test_visit_table.csv'
    adc_path = output_dir / 'test_adc_table.csv'
    
    print("\nSaving files...")
    patient_df.to_csv(patient_path, index=False)
    print(f"✓ {patient_path}")
    
    visit_df.to_csv(visit_path, index=False)
    print(f"✓ {visit_path}")
    
    adc_df.to_csv(adc_path, index=False)
    print(f"✓ {adc_path}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Patients: {len(patient_df):,}")
    print(f"Visits: {len(visit_df):,}")
    print(f"ADC Centers: {len(adc_df)}")
    print(f"Avg visits per patient: {len(visit_df)/len(patient_df):.1f}")
    
    # Print command to run TMV-HNN
    print("\n" + "=" * 80)
    print("✅ TEST DATA CREATED!")
    print("=" * 80)
    print("\nRun TMV-HNN with this command:\n")
    print(f"python3 run.py \\")
    print(f"    --patient_path {patient_path} \\")
    print(f"    --visit_path {visit_path} \\")
    print(f"    --adc_path {adc_path} \\")
    print(f"    --epochs_l3 10 \\")
    print(f"    --epochs_l2 10 \\")
    print(f"    --epochs_l1 10 \\")
    print(f"    --epochs_survival 10")
    print()


if __name__ == "__main__":
    main()
