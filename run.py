#!/usr/bin/env python3
"""
run.py - Main execution script for TMV-HNN model on NACC data

Usage:
    python3 run.py

This script:
1. Loads NACC normalized tables (patient, visit, ADC)
2. Preprocesses and extracts multi-modal features
3. Trains TMV-HNN (3-layer architecture)
4. Evaluates survival predictions
5. Generates results and visualizations
"""

import sys
import logging
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_loader import NACCDataLoader
from tmv_hnn import TMVHNNModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'tmv_hnn_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train TMV-HNN model on NACC data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--patient_path', type=str, 
                       default='output/nacc_patient_table_20251112_183501.csv',
                       help='Path to patient table CSV')
    
    parser.add_argument('--visit_path', type=str,
                       default='output/nacc_visit_table_20251112_183501.csv',
                       help='Path to visit table CSV')
    
    parser.add_argument('--adc_path', type=str,
                       default='output/nacc_adc_table_20251112_183501.csv',
                       help='Path to ADC table CSV')
    
    parser.add_argument('--output_dir', type=str, default='tmv_hnn_results/',
                       help='Output directory for results')
    
    parser.add_argument('--n_clusters', type=int, default=10,
                       help='Number of patient clusters (Layer 3)')
    
    parser.add_argument('--n_states', type=int, default=15,
                       help='Number of clinical states (Layer 2)')
    
    parser.add_argument('--latent_dim', type=int, default=64,
                       help='Latent dimension for Layer 3')
    
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for training')
    
    parser.add_argument('--epochs_l3', type=int, default=50,
                       help='Training epochs for Layer 3')
    
    parser.add_argument('--epochs_l2', type=int, default=30,
                       help='Training epochs for Layer 2')
    
    parser.add_argument('--epochs_l1', type=int, default=30,
                       help='Training epochs for Layer 1')
    
    parser.add_argument('--epochs_survival', type=int, default=50,
                       help='Training epochs for survival model')
    
    parser.add_argument('--max_patients', type=int, default=None,
                       help='Maximum number of patients to use (for testing)')
    
    return parser.parse_args()


def save_results(results: dict, model: TMVHNNModel, output_dir: Path):
    """
    Save results and model artifacts.
    
    Args:
        results: Dictionary with training results
        model: Trained TMV-HNN model
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cluster assignments
    logger.info("Saving cluster assignments...")
    
    cluster_df = pd.DataFrame({
        'patient_id': model.patient_ids,
        'cluster_l3': results['cluster_labels_l3'],
    })
    cluster_df.to_csv(output_dir / 'patient_clusters.csv', index=False)
    
    # Save embeddings
    logger.info("Saving patient embeddings...")
    embeddings_df = pd.DataFrame(
        results['final_embeddings'],
        index=model.patient_ids
    )
    embeddings_df.to_csv(output_dir / 'patient_embeddings.csv')
    
    # Save summary statistics
    logger.info("Saving summary statistics...")
    summary = {
        'n_patients': results['n_patients'],
        'n_clusters_l3': results['n_clusters_l3'],
        'n_states_l2': results['n_states_l2'],
        'training_date': datetime.now().isoformat()
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"Results saved to {output_dir}")


def generate_survival_predictions(model: TMVHNNModel, 
                                  output_dir: Path,
                                  n_samples: int = 10):
    """
    Generate and save survival predictions for sample patients.
    
    Args:
        model: Trained TMV-HNN model
        output_dir: Output directory
        n_samples: Number of sample patients
    """
    logger.info("Generating survival predictions for sample patients...")
    
    # Select random sample of patients
    n_patients = len(model.patient_ids)
    sample_indices = np.random.choice(n_patients, min(n_samples, n_patients), replace=False)
    
    # Evaluation times (years)
    eval_times = np.array([1, 2, 3, 4, 5])
    
    # Predict survival
    try:
        survival_probs, ci = model.predict_survival(
            patient_indices=sample_indices.tolist(),
            eval_times=eval_times,
            n_samples=100
        )
        
        # Save predictions
        predictions_df = pd.DataFrame(
            survival_probs,
            index=[model.patient_ids[i] for i in sample_indices],
            columns=[f'{t}yr' for t in eval_times]
        )
        predictions_df.to_csv(output_dir / 'survival_predictions.csv')
        
        # Save confidence intervals
        for i, t in enumerate(eval_times):
            ci_df = pd.DataFrame(
                ci[:, i, :],
                index=[model.patient_ids[j] for j in sample_indices],
                columns=['lower', 'upper']
            )
            ci_df.to_csv(output_dir / f'ci_{t}yr.csv')
            
        logger.info(f"Survival predictions saved for {len(sample_indices)} patients")
        
    except Exception as e:
        logger.warning(f"Could not generate survival predictions: {e}")


def print_summary(results: dict):
    """
    Print training summary.
    
    Args:
        results: Training results dictionary
    """
    print("\n" + "=" * 80)
    print("TMV-HNN TRAINING SUMMARY")
    print("=" * 80)
    
    print(f"\nDataset Statistics:")
    print(f"  Total patients: {results['n_patients']}")
    
    print(f"\nLayer 3 (Patient Profiles):")
    print(f"  Number of clusters: {results['n_clusters_l3']}")
    
    print(f"\nLayer 2 (Clinical States):")
    print(f"  Number of states: {results['n_states_l2']}")
    
    print(f"\nLayer 1 (Hypergraph):")
    if 'n_hyperedges' in results:
        print(f"  Number of hyperedges: {results.get('n_hyperedges', 'N/A')}")
    
    print("\n" + "=" * 80)


def main():
    """Main execution function."""
    args = parse_arguments()
    
    start_time = datetime.now()
    
    logger.info("=" * 80)
    logger.info("TMV-HNN: Temporal Multi-View Hypergraph Neural Network")
    logger.info("For Alzheimer's Disease Progression Prediction")
    logger.info("=" * 80)
    logger.info(f"Start time: {start_time}")
    logger.info(f"Configuration:")
    logger.info(f"  Patient clusters (K_3): {args.n_clusters}")
    logger.info(f"  Clinical states (K_2): {args.n_states}")
    logger.info(f"  Latent dimension: {args.latent_dim}")
    logger.info(f"  Device: {args.device}")
    
    try:
        # 1. Load data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Loading NACC Data")
        logger.info("=" * 80)
        
        data_loader = NACCDataLoader(
            patient_path=args.patient_path,
            visit_path=args.visit_path,
            adc_path=args.adc_path
        )
        
        data = data_loader.load_data()
        patient_df = data['patient']
        visit_df = data['visit']
        
        # Subsample if requested (for testing)
        if args.max_patients:
            logger.info(f"Subsampling to {args.max_patients} patients for testing...")
            patient_ids = patient_df['NACCID'].unique()[:args.max_patients]
            patient_df = patient_df[patient_df['NACCID'].isin(patient_ids)]
            visit_df = visit_df[visit_df['NACCID'].isin(patient_ids)]
        
        # 2. Extract features
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Extracting Multi-Modal Features")
        logger.info("=" * 80)
        
        patient_data = data_loader.extract_modalities(patient_df, visit_df)
        
        # 3. Prepare datasets
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Preparing Datasets")
        logger.info("=" * 80)
        
        # Baseline features for Layer 3
        baseline_features, modality_mask, patient_ids = data_loader.create_baseline_dataset(patient_data)
        logger.info(f"Baseline features shape: {baseline_features.shape}")
        
        # Sequences for Layer 2
        sequences = data_loader.create_sequence_dataset(patient_data)
        logger.info(f"Created sequences for {len(sequences)} patients")
        
        # Infer sequence input dimension
        sample_seq = next(iter(sequences.values()))
        sequence_input_dim = sample_seq['features'].shape[1]
        logger.info(f"Sequence feature dimension: {sequence_input_dim}")
        
        # Survival data
        survival_data = data_loader.create_survival_data(patient_df, visit_df)
        n_events = sum(1 for v in survival_data.values() if v['event'] == 1)
        logger.info(f"Survival data: {len(survival_data)} patients, {n_events} events")
        
        # 4. Initialize and train TMV-HNN
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Training TMV-HNN Model")
        logger.info("=" * 80)
        
        model = TMVHNNModel(
            n_clusters_l3=args.n_clusters,
            n_states_l2=args.n_states,
            latent_dim=args.latent_dim,
            hidden_dim_l2=128,
            hidden_dims_l1=[128, 64],
            survival_hidden_dims=[128, 64],
            device=args.device
        )
        
        results = model.fit(
            baseline_features=baseline_features,
            modality_mask=modality_mask,
            sequences=sequences,
            survival_data=survival_data,
            patient_ids=patient_ids,
            sequence_input_dim=sequence_input_dim,
            n_epochs_l3=args.epochs_l3,
            n_epochs_l2=args.epochs_l2,
            n_epochs_l1=args.epochs_l1,
            n_epochs_survival=args.epochs_survival
        )
        
        # 5. Save results
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Saving Results")
        logger.info("=" * 80)
        
        output_dir = Path(args.output_dir)
        save_results(results, model, output_dir)
        
        # 6. Generate predictions
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Generating Survival Predictions")
        logger.info("=" * 80)
        
        generate_survival_predictions(model, output_dir, n_samples=10)
        
        # 7. Print summary
        print_summary(results)
        
        # Final statistics
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"End time: {end_time}")
        logger.info(f"Total duration: {duration}")
        logger.info(f"Results saved to: {output_dir.absolute()}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
