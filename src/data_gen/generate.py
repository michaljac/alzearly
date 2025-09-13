#!/usr/bin/env python3
"""
Command-line interface for synthetic patient-year data generation.
Implements R-DATA-001 v1.0 requirements.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from modules to avoid circular imports
import importlib.util
import os

# Load data_gen module directly
data_gen_path = os.path.join(os.path.dirname(__file__), '..', 'data_gen.py')
spec = importlib.util.spec_from_file_location("data_gen_module", data_gen_path)
data_gen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_gen_module)
SyntheticDataGenerator = data_gen_module.SyntheticDataGenerator

from src.config import load_config


def main():
    """Main CLI entry point for data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic patient-year data (R-DATA-001 v1.0)"
    )
    
    parser.add_argument(
        "--patients", 
        type=int, 
        default=200000,
        help="Number of patients to generate (default: 200000)"
    )
    
    parser.add_argument(
        "--years",
        type=int, 
        default=10,
        help="Number of years to generate (default: 10)"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        default="/Data/raw",
        help="Output directory for partitioned Parquet files"
    )
    
    parser.add_argument(
        "--prevalence",
        type=float,
        default=0.08,
        help="Alzheimer's prevalence rate (default: 0.08)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/data_gen.yaml",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Generate tiny dataset for fast CI (1000 patients, 3 years)"
    )
    
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Don't clean existing data (append mode)"
    )
    
    args = parser.parse_args()
    
    # Load base configuration from config/data_gen.yaml
    try:
        config = load_config("data_gen", args.config)
    except FileNotFoundError:
        print(f"ERROR: Config file {args.config} not found. Please ensure config/data_gen.yaml exists.")
        sys.exit(1)
    
    # Override with command line arguments only when explicitly provided
    if args.tiny:
        # Tiny mode overrides for fast testing
        config['dataset']['n_patients'] = 1000
        config['dataset']['years'] = [2022, 2023, 2024]
        config['processing']['rows_per_chunk'] = 1000
    else:
        # Only override patients if explicitly provided (not default 200000)
        if args.patients != 200000:
            config['dataset']['n_patients'] = args.patients
        
        # Only override years if explicitly provided (not default 10)
        if args.years != 10:
            config['dataset']['years'] = list(range(2024 - args.years + 1, 2025))
    
    # Only override these if explicitly provided (not defaults)
    if args.prevalence != 0.08:
        config['target']['positive_rate'] = args.prevalence
    if args.seed != 42:
        config['processing']['seed'] = args.seed
    if args.out != "/Data/raw":
        config['output']['directory'] = args.out
    
    # Always set clean flag
    config['output']['clean_existing'] = not args.no_clean
    
    # Check for existing data and prompt user (unless --no-clean, --tiny, or running in Docker)
    output_path = Path(config['output']['directory'])
    is_docker = os.environ.get('PYTHONPATH') == '/workspace'  # Docker container indicator
    
    if not args.no_clean and not args.tiny and not is_docker and output_path.exists():
        # Check if there are any year directories
        year_dirs = [d for d in output_path.iterdir() if d.is_dir() and (d.name.startswith('year=') or d.name.isdigit())]
        if year_dirs:
            print(f"Existing data found in {output_path}")
            year_names = [d.name for d in year_dirs[:3]]
            suffix = '...' if len(year_dirs) > 3 else ''
            print(f"Found {len(year_dirs)} year directories: {year_names}{suffix}")
            
            while True:
                response = input("Regenerate data? This will delete existing files (y/n): ").strip().lower()
                if response in ['y', 'yes']:
                    print("Proceeding with data generation...")
                    break
                elif response in ['n', 'no']:
                    print("Using existing data. Exiting.")
                    sys.exit(0)
                else:
                    print("Please enter 'y' or 'n'")
    
    # Create generator and run
    generator = SyntheticDataGenerator(config)
    
    generator.generate()



if __name__ == "__main__":
    main()
