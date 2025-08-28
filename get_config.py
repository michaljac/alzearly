#!/usr/bin/env python3
"""
Get configuration parameters for train.bat
Uses the existing src/config.py system
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import load_config

def main():
    """Load data_gen config and output parameters for train.bat."""
    try:
        config = load_config("data_gen")
        
        # Output parameters in batch-compatible format
        print(f"set CONFIG_N_PATIENTS={config.n_patients}")
        print(f"set CONFIG_YEARS={','.join(map(str, config.years))}")
        print(f"set CONFIG_POSITIVE_RATE={config.positive_rate}")
        print(f"set CONFIG_SEED={config.seed}")
        print(f"set CONFIG_ROWS_PER_CHUNK={config.rows_per_chunk}")
        print(f"set CONFIG_OUTPUT_DIR={config.output_dir}")
        print(f"set CONFIG_TARGET_COLUMN={config.target_column}")
        
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        # Set default values
        print("set CONFIG_N_PATIENTS=3000")
        print("set CONFIG_YEARS=2021,2022,2023,2024")
        print("set CONFIG_POSITIVE_RATE=0.08")
        print("set CONFIG_SEED=42")
        print("set CONFIG_ROWS_PER_CHUNK=100000")
        print("set CONFIG_OUTPUT_DIR=data/raw")
        print("set CONFIG_TARGET_COLUMN=alzheimers_diagnosis")
        sys.exit(1)

if __name__ == "__main__":
    main()
