#!/usr/bin/env python3
"""
Data generation CLI for AlzEarly project.

Generates synthetic patient data and preprocesses it for model training.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---- Permissions -------------------------------------------------------------
DIR_MODE = 0o777
FILE_MODE = 0o666

DATA_ROOT = Path("/Data")
RAW_DIR   = DATA_ROOT / "raw"
FEAT_DIR  = DATA_ROOT / "featurized"


def set_permissive_umask():
    try:
        os.umask(0)
    except OSError as e:
        print(f"WARNING: could not set umask(0): {e}")


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p, DIR_MODE)
    except OSError as e:
        print(f"WARNING: chmod {p} -> {oct(DIR_MODE)} failed: {e}")


def ensure_tree():
    ensure_dir(DATA_ROOT)
    ensure_dir(RAW_DIR)
    ensure_dir(FEAT_DIR)


def chmod_recursive(root):
    try:
        if root.is_dir():
            os.chmod(root, DIR_MODE)
            for base, dirs, files in os.walk(root):
                for d in dirs:
                    try: os.chmod(Path(base) / d, DIR_MODE)
                    except OSError as e: print(f"WARNING: chmod dir {Path(base)/d}: {e}")
                for f in files:
                    try: os.chmod(Path(base) / f, FILE_MODE)
                    except OSError as e: print(f"WARNING: chmod file {Path(base)/f}: {e}")
        elif root.exists():
            os.chmod(root, FILE_MODE)
    except OSError as e:
        print(f"WARNING: recursive chmod at {root}: {e}")


# ---- Setup / checks ----------------------------------------------------------
def setup_paths():
    root = Path(__file__).parent.resolve()
    # Navigate to project root (two levels up from src/cli/)
    project_root = root.parent.parent
    src = project_root / "src"
    
    # Add both project root and src to sys.path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    
    print(f"Project root: {project_root}")
    print(f"Source directory: {src}")
    return True


def featurized_exists():
    if FEAT_DIR.exists():
        if list(FEAT_DIR.glob("*.parquet")) or list(FEAT_DIR.glob("*.csv")):
            print(f"Found featurized data in {FEAT_DIR}")
            return True
    return False


# ---- Pipeline ----------------------------------------------------------------
def run_pipeline(args):
    print("Alzearly Data Generation")
    print("=======================================")

    set_permissive_umask()
    ensure_tree()

    # Check if modules exist (we'll call them via subprocess)
    try:
        import src.data_gen
        import src.preprocess
    except ImportError as e:
        print(f"ERROR: Failed to import required modules: {e}")
        print("Make sure src/data_gen.py and src/preprocess.py exist")
        return 1

    # Raw output dir
    raw_out = Path(args.output_dir or str(RAW_DIR))
    # Normalize accidental /Data/alzearly/... back to /Data/...
    s = str(raw_out)
    if s == "/Data/alzearly":
        raw_out = DATA_ROOT
    elif s.startswith("/Data/alzearly/"):
        raw_out = DATA_ROOT / Path(s).relative_to("/Data/alzearly")
    ensure_dir(raw_out)

    # Early exit if data exists and not forcing
    if featurized_exists() and not args.force_regen:
        print("Featurized data already exists. Use --force-regen to regenerate.")
        return 0

    # Data generation
    if not args.skip_data_gen:
        print("Step 1: Data Generation")
        try:
            # Use direct import instead of subprocess
            from src.data_gen import SyntheticDataGenerator
            from src.config import load_config
            
            # Load config and set parameters
            config = load_config("data_gen")
            config['dataset']['n_patients'] = args.num_patients or config['dataset']['n_patients']
            config['processing']['seed'] = args.seed or config['processing']['seed']
            config['output']['directory'] = str(raw_out)
            config['output']['clean_existing'] = args.force_regen
            
            # Generate data
            generator = SyntheticDataGenerator(config)
            generator.generate()
                
            print("Data generation: done.")
        except Exception as e:
            print(f"ERROR: data generation failed: {e}")
            return 1
    else:
        print("Skipping data generation (--skip-data-gen).")

    # Preprocessing
    if not args.skip_preprocess:
        print("Step 2: Preprocessing")
        try:
            # Call preprocess function through subprocess as CLI command
            cmd = [
                sys.executable, "-m", "src.preprocess",
                "--input-dir", str(raw_out),
                "--output-dir", str(FEAT_DIR),
                "--rolling-window", "3",
                "--chunk-size", "3000",
                "--seed", str(args.seed or 42)
            ]
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"ERROR: Command failed with exit code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return 1
                
            print("Preprocessing: done.")
        except (subprocess.SubprocessError, OSError) as e:
            print(f"ERROR: preprocessing failed: {e}")
            return 1
    else:
        print("Skipping preprocessing (--skip-preprocess).")

    # Final: recursive perms under /Data
    print("Fixing permissions under /Data ...")
    for p in (DATA_ROOT, raw_out, FEAT_DIR):
        chmod_recursive(p)
    print("All done.")
    return 0


# ---- CLI ---------------------------------------------------------------------
def main():
    if not setup_paths():
        return 1

    # Load config file for defaults
    from src.config import load_config
    config = load_config("data_gen")
    
    ap = argparse.ArgumentParser(description="Alzearly Data Generation")
    ap.add_argument("--num-patients", type=int, default=config['dataset']['n_patients'])
    ap.add_argument("--output-dir", type=str, default=str(RAW_DIR))
    ap.add_argument("--seed", type=int, default=config['processing']['seed'])
    ap.add_argument("--force-regen", action="store_true")
    ap.add_argument("--skip-data-gen", action="store_true")
    ap.add_argument("--skip-preprocess", action="store_true", default=False)
    args = ap.parse_args()

    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
