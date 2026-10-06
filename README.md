# Three-Layer Alzheimer Progression Model

A research-oriented Python pipeline for building and evaluating a temporal multi-view hypergraph neural network (TMV-HNN) for Alzheimer's disease progression using NACC (National Alzheimer's Coordinating Center) data.

This repository combines two complementary pieces:

- A data processing pipeline that normalizes UDS and MRI tables into a three-table patient/visit/ADC structure.
- A modeling pipeline that extracts multimodal features, learns patient clusters and clinical states, builds a hypergraph representation, and performs survival analysis for disease progression.

The project is designed for exploratory biomedical modeling rather than clinical deployment.

## Overview

The repository implements a 3-layer architecture:

1. Layer 3: patient profile clustering from baseline multimodal features
2. Layer 2: temporal clinical-state modeling across visits
3. Layer 1: hypergraph-based patient interaction modeling

The final stage integrates these representations into a survival model that estimates risk and progression over time.

## Features

- NACC data ingestion and preprocessing
- Normalized three-table dataset generation (`patient`, `visit`, `adc`)
- Multimodal feature extraction from:
  - demographics
  - genetics
  - cognitive assessments
  - symptom and functional measures
  - MRI-derived variables
- Temporal sequence generation for longitudinal progression
- Patient clustering and latent embedding learning
- Clinical state discovery across visits
- Hypergraph-based representation learning
- Bayesian neural survival modeling for progression risk prediction
- Result export to CSV/JSON for downstream analysis

## Repository Structure

```text
.
├── auto_run.py
├── create_test_data.py
├── data_loader.py
├── find_nacc_data.py
├── layer1_hypergraph.py
├── layer2_clinical_states.py
├── layer3_patient_clustering.py
├── run.py
├── survival_model.py
├── tmv_hnn.py
├── update_data_loader.py
├── verify_data_structure.py
├── data-processing/
│   ├── data_quality.py
│   ├── example_usage.py
│   ├── nacc_processing_fixed.py
│   ├── nacc_processor.py
│   ├── run.py
│   └── utils.py
└── output/                     # generated datasets/results (when created)
```

## Data Pipeline

The `data-processing/` directory contains the preprocessing and normalization logic for NACC datasets. It is designed to take UDS clinical data and optional MRI/SCAN data and produce a consistent three-table structure.

Typical workflow:

```bash
python data-processing/run.py \
  --uds_path path/to/uds_data.csv \
  --mri_path path/to/scan_mri.csv \
  --output_dir output
```

This step produces normalized tables such as:

- `patient` table
- `visit` table
- `adc` table
- optional baseline and summary datasets

## Modeling Pipeline

Once the normalized NACC tables are available, the main training script loads them and trains the TMV-HNN model.

```bash
python run.py \
  --patient_path output/nacc_patient_table_20251112_183501.csv \
  --visit_path output/nacc_visit_table_20251112_183501.csv \
  --adc_path output/nacc_adc_table_20251112_183501.csv \
  --output_dir tmv_hnn_results
```

The training script performs the following:

1. Loads patient, visit, and ADC tables
2. Extracts multimodal baseline and temporal features
3. Creates baseline dataset for patient clustering
4. Creates longitudinal sequence data for clinical states
5. Trains the layered TMV-HNN model
6. Trains a survival model for progression risk
7. Saves embeddings, cluster assignments, and survival predictions

## Outputs

The model saves outputs into the configured output directory (for example, `tmv_hnn_results/`), including:

- `patient_clusters.csv`
- `patient_embeddings.csv`
- `summary.json`
- `survival_predictions.csv`
- `ci_<time>yr.csv`

## Requirements

This project is implemented in Python and relies on standard scientific computing libraries:

```bash
pip install pandas numpy scikit-learn torch
```

Depending on the environment and dataset format, additional packages may be required for CSV export or optional statistical workflows.

## Usage Notes

- This is a research prototype and not intended for clinical diagnosis.
- Output files are generated in a format suitable for downstream analysis, not necessarily for a production API.
- The model expects patient identifiers and NACC-style tables with consistent field names.
- For MRI matching, the preprocessing pipeline uses temporal proximity to connect MRI observations with visit records.

## Recommended Workflow

```bash
# 1. Prepare data
python data-processing/run.py --uds_path data/uds_data.csv --mri_path data/scan_mri.csv --output_dir output

# 2. Train model
python run.py --patient_path output/nacc_patient_table_20251112_183501.csv \
             --visit_path output/nacc_visit_table_20251112_183501.csv \
             --adc_path output/nacc_adc_table_20251112_183501.csv \
             --output_dir tmv_hnn_results
```

## Notes

This repository is strongly tied to NACC data conventions and disease progression modeling. For best results, use curated and validated clinical datasets with a consistent schema before training the model.

## License

No explicit license file is present in the repository at this time. If you plan to reuse or distribute this code, confirm the licensing terms with the repository owner before publication or commercial use.

## Acknowledgements

This project is built around NACC data and Alzheimer’s disease progression modeling research. It is intended for exploratory, academic, and experimental use.
