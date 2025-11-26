#!/usr/bin/env python3
"""
run.py - Main execution script for NACC dataset processing

This script implements the dataset preparation described in the LaTeX document,
creating a normalized three-table structure from NACC UDS and SCAN MRI data.

Usage:
    python3 run.py --uds_path <path_to_uds_csv> [--mri_path <path_to_mri_csv>] [options]

Example:
    python3 run.py --uds_path data/uds_data.csv --mri_path data/scan_mri.csv --output_dir results/
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

# Import our modules
from nacc_processor import NACCDataProcessor
from data_quality import DataQualityChecker, validate_nacc_data
from utils import (
    save_datasets, 
    generate_codebook, 
    create_summary_statistics,
    create_baseline_dataset,
    export_for_statistical_software,
    print_processing_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nacc_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Process NACC UDS and MRI datasets to create normalized three-table structure.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with UDS data only
  python3 run.py --uds_path data/uds_data.csv
  
  # Include MRI data
  python3 run.py --uds_path data/uds_data.csv --mri_path data/scan_mri.csv
  
  # Specify output directory and format
  python3 run.py --uds_path data/uds_data.csv --output_dir results/ --format parquet
  
  # Change temporal matching threshold for MRI
  python3 run.py --uds_path data/uds_data.csv --mri_path data/scan_mri.csv --temporal_threshold 60
        """
    )
    
    # Required arguments
    parser.add_argument('--uds_path', type=str, required=True,
                       help='Path to UDS clinical dataset CSV file')
    
    # Optional arguments
    parser.add_argument('--mri_path', type=str, default=None,
                       help='Path to SCAN MRI dataset CSV file (optional)')
    
    parser.add_argument('--output_dir', type=str, default='output/',
                       help='Directory to save output files (default: output/)')
    
    parser.add_argument('--temporal_threshold', type=int, default=90,
                       help='Days threshold for matching MRI to visits (default: 90)')
    
    parser.add_argument('--format', type=str, default='csv',
                       choices=['csv', 'parquet', 'stata', 'excel'],
                       help='Output file format (default: csv)')
    
    parser.add_argument('--compression', type=str, default=None,
                       choices=['gzip', 'bz2', 'zip', 'xz', None],
                       help='Compression for output files (default: None)')
    
    parser.add_argument('--create_baseline', action='store_true',
                       help='Also create a baseline-only dataset')
    
    parser.add_argument('--export_stats', action='store_true',
                       help='Export datasets formatted for statistical software (R, SAS)')
    
    parser.add_argument('--skip_quality', action='store_true',
                       help='Skip data quality validation')
    
    parser.add_argument('--skip_codebook', action='store_true',
                       help='Skip codebook generation')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Log start of processing
    start_time = datetime.now()
    logger.info("="*80)
    logger.info("NACC Dataset Processing Started")
    logger.info(f"Start time: {start_time}")
    logger.info(f"UDS path: {args.uds_path}")
    logger.info(f"MRI path: {args.mri_path if args.mri_path else 'Not provided'}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Temporal threshold: {args.temporal_threshold} days")
    logger.info("="*80)
    
    try:
        # Initialize processor
        processor = NACCDataProcessor(
            uds_path=args.uds_path,
            mri_path=args.mri_path,
            temporal_threshold=args.temporal_threshold
        )
        
        # Process datasets
        logger.info("\n*** Step 1: Processing datasets ***")
        datasets = processor.process_datasets()
        
        # Data quality validation
        quality_report = None
        if not args.skip_quality:
            logger.info("\n*** Step 2: Running data quality validation ***")
            quality_report = validate_nacc_data(
                datasets['patient'],
                datasets['visit'],
                datasets['adc']
            )
            
            # Save quality report
            quality_dir = Path(args.output_dir) / 'quality_reports'
            quality_dir.mkdir(parents=True, exist_ok=True)
            
            for report_name, report_df in quality_report.items():
                report_path = quality_dir / f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                report_df.to_csv(report_path, index=False)
                logger.info(f"Saved {report_name} to {report_path}")
        else:
            logger.info("\n*** Step 2: Skipping data quality validation ***")
            
        # Generate summary statistics
        logger.info("\n*** Step 3: Generating summary statistics ***")
        summary_stats = create_summary_statistics(datasets)
        
        # Save summary statistics
        stats_dir = Path(args.output_dir) / 'summary_stats'
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        for stat_name, stat_df in summary_stats.items():
            stat_path = stats_dir / f"{stat_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            stat_df.to_csv(stat_path)
            logger.info(f"Saved {stat_name} to {stat_path}")
            
        # Create baseline dataset if requested
        if args.create_baseline:
            logger.info("\n*** Step 4: Creating baseline dataset ***")
            baseline_df = create_baseline_dataset(datasets['patient'], datasets['visit'])
            datasets['baseline'] = baseline_df
            
        # Save main datasets
        logger.info(f"\n*** Step 5: Saving datasets in {args.format} format ***")
        save_datasets(
            datasets,
            args.output_dir,
            format=args.format,
            compression=args.compression
        )
        
        # Generate codebook
        if not args.skip_codebook:
            logger.info("\n*** Step 6: Generating codebook ***")
            codebook_path = Path(args.output_dir) / f"codebook_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            generate_codebook(datasets, codebook_path)
        else:
            logger.info("\n*** Step 6: Skipping codebook generation ***")
            
        # Export for statistical software if requested
        if args.export_stats:
            logger.info("\n*** Step 7: Exporting for statistical software ***")
            export_dir = Path(args.output_dir) / 'statistical_exports'
            export_for_statistical_software(datasets, export_dir)
            
        # Print summary
        print_processing_summary(datasets, quality_report)
        
        # Save processing metadata
        metadata = {
            'processing_date': datetime.now().isoformat(),
            'uds_path': args.uds_path,
            'mri_path': args.mri_path,
            'temporal_threshold': args.temporal_threshold,
            'output_format': args.format,
            'compression': args.compression,
            'datasets_created': list(datasets.keys()),
            'processing_time_seconds': (datetime.now() - start_time).total_seconds()
        }
        
        metadata_path = Path(args.output_dir) / 'processing_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Log completion
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info("\n" + "="*80)
        logger.info("NACC Dataset Processing Completed Successfully!")
        logger.info(f"End time: {end_time}")
        logger.info(f"Total duration: {duration}")
        logger.info(f"Output saved to: {Path(args.output_dir).absolute()}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()