#!/usr/bin/env python3

import argparse
import os
import sys
import subprocess
import importlib
from pathlib import Path

# ---- Permissions -------------------------------------------------------------
DIR_MODE = 0o777
FILE_MODE = 0o666

DATA_ROOT = Path("/Data")
RAW_DIR   = DATA_ROOT / "raw"
FEAT_DIR  = DATA_ROOT / "featurized"


def set_permissive_umask() -> None:
    try:
        os.umask(0)
    except Exception as e:
        print(f"WARNING: could not set umask(0): {e}")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p, DIR_MODE)
    except Exception as e:
        print(f"WARNING: chmod {p} -> {oct(DIR_MODE)} failed: {e}")


def ensure_tree() -> None:
    ensure_dir(DATA_ROOT)
    ensure_dir(RAW_DIR)
    ensure_dir(FEAT_DIR)


def chmod_recursive(root: Path) -> None:
    try:
        if root.is_dir():
            os.chmod(root, DIR_MODE)
            for base, dirs, files in os.walk(root):
                for d in dirs:
                    try: os.chmod(Path(base) / d, DIR_MODE)
                    except Exception as e: print(f"WARNING: chmod dir {Path(base)/d}: {e}")
                for f in files:
                    try: os.chmod(Path(base) / f, FILE_MODE)
                    except Exception as e: print(f"WARNING: chmod file {Path(base)/f}: {e}")
        elif root.exists():
            os.chmod(root, FILE_MODE)
    except Exception as e:
        print(f"WARNING: recursive chmod at {root}: {e}")


# ---- Setup / checks ----------------------------------------------------------
def setup_paths() -> bool:
    root = Path(__file__).parent.resolve()
    src = root / "src"
    # Add both project root and src to sys.path
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if not src.exists():
        print(f"ERROR: src directory not found at {src}")
        return False
    return True


def check_dependencies() -> bool:
    required = ["polars", "pyarrow", "faker", "pandas", "numpy"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Installing requirements-datagen.txt ...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements-datagen.txt"],
                check=True, capture_output=True, text=True
            )
            print("Dependencies installed.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: pip install failed: {e}")
            return False
    return True


def featurized_exists() -> bool:
    if FEAT_DIR.exists():
        if list(FEAT_DIR.glob("*.parquet")) or list(FEAT_DIR.glob("*.csv")):
            print(f"Found featurized data in {FEAT_DIR}")
            return True
    return False


def try_import_generate():
    """
    Try multiple import patterns for the generate function:
    - src.data_gen: module or package
    - data_gen:     module or package
    Returns a callable or raises ImportError with a helpful message.
    """
    candidates = [
        ("src.data_gen", "generate"),
        ("src.data_gen.generate", "generate"),
        ("data_gen", "generate"),
        ("data_gen.generate", "generate"),
    ]
    last_err = None
    for modname, attr in candidates:
        try:
            mod = importlib.import_module(modname)
            fn = getattr(mod, attr, None)
            if callable(fn):
                return fn
        except Exception as e:
            last_err = e
    raise ImportError(
        "Could not import a 'generate' function. "
        "Create one of:\n"
        "  - src/data_gen.py  (with def generate(...))\n"
        "  - src/data_gen/__init__.py or /generate.py (with def generate(...))\n"
        "  - data_gen.py or data_gen/ (with def generate(...))\n"
        f"Last error: {last_err}"
    )


def try_import_preprocess():
    """
    Try multiple import patterns for the preprocess function:
    - src.preprocess: module or package
    - preprocess:     module or package
    Returns a callable or raises ImportError with a helpful message.
    """
    candidates = [
        ("src.preprocess", "preprocess"),
        ("src.preprocess.preprocess", "preprocess"),
        ("preprocess", "preprocess"),
        ("preprocess.preprocess", "preprocess"),
    ]
    last_err = None
    for modname, attr in candidates:
        try:
            mod = importlib.import_module(modname)
            fn = getattr(mod, attr, None)
            if callable(fn):
                return fn
        except Exception as e:
            last_err = e
    raise ImportError(
        "Could not import a 'preprocess' function. "
        "Create one of:\n"
        "  - src/preprocess.py (with def preprocess(...))\n"
        "  - src/preprocess/__init__.py or /preprocess.py (with def preprocess(...))\n"
        "  - preprocess.py or preprocess/ (with def preprocess(...))\n"
        f"Last error: {last_err}"
    )


# ---- Pipeline ----------------------------------------------------------------
def run_pipeline(args) -> int:
    print("Alzearly Data Generation")
    print("=======================================")

    set_permissive_umask()
    ensure_tree()

    # Optional config
    load_config = None
    try:
        load_config = importlib.import_module("config").load_config
    except Exception:
        try:
            load_config = importlib.import_module("src.config").load_config
        except Exception:
            load_config = None

    try:
        generate_data = try_import_generate()
        preprocess_data = try_import_preprocess()
    except ImportError as e:
        print(f"ERROR: {e}")
        return 1

    # Defaults from config if available
    default_n = 3000
    default_seed = 42
    if load_config:
        try:
            cfg = load_config("data_gen")
            default_n = getattr(cfg, "n_patients", default_n)
            default_seed = getattr(cfg, "seed", default_seed)
        except Exception as e:
            print(f"NOTE: could not load defaults from config: {e}")

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
        years, positive_rate = None, None
        if load_config:
            try:
                cfg = load_config("data_gen")
                years = getattr(cfg, "years", None)
                positive_rate = getattr(cfg, "positive_rate", None)
            except Exception as e:
                print(f"NOTE: could not load extra params from config: {e}")
        years = years or [2021, 2022, 2023, 2024]
        positive_rate = positive_rate or 0.08

        try:
            generate_data(
                config_file=None,
                n_patients=args.num_patients or default_n,
                years=",".join(map(str, years)),
                positive_rate=positive_rate,
                out=str(raw_out),
                seed=args.seed or default_seed,
            )
            print("Data generation: done.")
        except Exception as e:
            print(f"ERROR: data generation failed: {e}")
            return 1
    else:
        print("Skipping data generation (--skip-data-gen).")

    # Preprocessing
    if not args.skip_preprocess:
        print("Step 2: Preprocessing")
        rolling_window_years = 3
        chunk_size = 3000
        if load_config:
            try:
                cfg = load_config("data_gen")
                chunk_size = getattr(cfg, "rows_per_chunk", chunk_size)
            except Exception as e:
                print(f"NOTE: could not load preprocess params: {e}")

        try:
            preprocess_data(
                config_file=None,
                input_dir=str(raw_out),
                output_dir=str(FEAT_DIR),
                rolling_window_years=rolling_window_years,
                chunk_size=chunk_size,
                seed=args.seed or default_seed,
            )
            print("Preprocessing: done.")
        except Exception as e:
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
def main() -> int:
    if not setup_paths():
        return 1
    if not check_dependencies():
        return 1

    ap = argparse.ArgumentParser(description="Alzearly Data Generation")
    ap.add_argument("--num-patients", type=int, default=None)
    ap.add_argument("--output-dir", type=str, default=str(RAW_DIR))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force-regen", action="store_true")
    ap.add_argument("--skip-data-gen", action="store_true")
    ap.add_argument("--skip-preprocess", action="store_true", default=False)
    args = ap.parse_args()

    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
