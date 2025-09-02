"""
Utility functions for the Alzearly pipeline.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Any

def set_seed(seed: int) -> None:
    """Set random seeds for deterministic runs across all libraries."""
    import random
    import numpy as np
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set XGBoost seed if available
    try:
        import xgboost as xgb
        # XGBoost doesn't have a global set_random_state, but we can set it in model params
        # The seed will be used when creating XGBoost models
        pass
    except ImportError:
        pass
    
    # Set LightGBM seed if available
    try:
        import lightgbm as lgb
        lgb.set_random_state(seed)
    except ImportError:
        pass
    
    # Set scikit-learn seed
    try:
        from sklearn.utils import check_random_state
        check_random_state(seed)
    except ImportError:
        pass
    
    print(f"🌱 Random seed set to: {seed}")

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    import logging
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_input(prompt: str, input_type: str = "text", allow_quit: bool = False, valid_choices: Optional[List[str]] = None) -> str:
    """Get user input with validation."""
    while True:
        try:
            user_input = input(prompt).strip()
            
            if allow_quit and user_input.lower() in ['q', 'quit', 'exit']:
                return 'q'
            
            if input_type == "y/n":
                if user_input.lower() in ['y', 'yes', 'n', 'no']:
                    return user_input.lower()
                print("❌ Please enter 'y' or 'n'")
                continue
            
            if input_type == "choice" and valid_choices:
                if user_input in valid_choices:
                    return user_input
                print(f"❌ Invalid choice. Please enter one of: {', '.join(valid_choices)}")
                continue
            
            return user_input
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)
        except EOFError:
            print("\n👋 Goodbye!")
            sys.exit(0)

def check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    # Map package names to their import names
    package_imports = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'scikit-learn': 'sklearn',
        'xgboost': 'xgboost',
        'polars': 'polars',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'tqdm': 'tqdm',
        'typer': 'typer',
        'mlflow': 'mlflow'
    }
    
    missing_packages = []
    
    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are available")
    return True

def create_directories(directories: List[str]) -> None:
    """Create directories if they don't exist."""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def handle_critical_error(error: Exception, message: str = "Critical error occurred") -> None:
    """Handle critical errors."""
    print(f"❌ {message}: {error}")
    sys.exit(1)

def handle_recoverable_error(error: Exception, message: str = "Error occurred") -> bool:
    """Handle recoverable errors."""
    print(f"⚠️  {message}: {error}")
    return False


# =============================================================================
# EXPERIMENT TRACKING FUNCTIONS
# =============================================================================

def setup_experiment_tracker():
    """
    Bootstrap function to set up experiment tracking.
    Returns (tracker, tracker_type) tuple.
    """
    # Suppress Pydantic warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    warnings.filterwarnings("ignore", message="Field.*has conflict with protected namespace")
    warnings.filterwarnings("ignore", message="Valid config keys have changed in V2")
    
    # Check if we're in non-interactive mode (Docker container)
    import os
    if os.environ.get('NON_INTERACTIVE', 'false').lower() == 'true':
        print("\n🔬 Experiment Tracking Setup (Non-interactive mode)")
        print("=" * 50)
        print("✅ Using no experiment tracking for Docker container")
        return None, "none"
    
    # Check if we're in a non-interactive environment
    try:
        import sys
        if not sys.stdin.isatty():
            print("\n🔬 Experiment Tracking Setup (Non-interactive environment)")
            print("=" * 50)
            print("✅ Using no experiment tracking for non-interactive environment")
            return None, "none"
    except:
        # If we can't check, assume non-interactive
        print("\n🔬 Experiment Tracking Setup (Non-interactive environment)")
        print("=" * 50)
        print("✅ Using no experiment tracking for non-interactive environment")
        return None, "none"
    
    print("\n🔬 Experiment Tracking Setup")
    print("=" * 50)
    
    # Present menu to user
    print("Select experiment tracker:")
    print("1. MLflow (local)")
    print("2. No tracking")
    
    while True:
        try:
            choice = input("\nEnter choice (1-2, default=1): ").strip()
            if choice == "":
                choice = "1"
            
            if choice in ["1", "2"]:
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2")
                
        except (KeyboardInterrupt, EOFError, OSError) as e:
            print(f"\n\n⚠️  Input error ({type(e).__name__}). Using no tracking.")
            print("💡 This might happen in non-interactive environments.")
            return None, "none"
    
    # Handle MLflow selection
    if choice == "1":
        return setup_mlflow()
    
    # Handle no tracking
    else:
        print("✅ No experiment tracking will be used.")
        return None, "none"



def setup_mlflow():
    """Set up MLflow tracking."""
    print("\n🔧 Setting up MLflow...")
    
    # Suppress Pydantic warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    warnings.filterwarnings("ignore", message="Field.*has conflict with protected namespace")
    warnings.filterwarnings("ignore", message="Valid config keys have changed in V2")
    
    # Try to import mlflow
    try:
        import mlflow
        print("✅ mlflow available")
    except ImportError:
        print("❌ mlflow not available - please install it in the Dockerfile")
        print("🔄 Falling back to no tracking")
        return None, "none"
    
    # Set up MLflow tracking
    try:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("alzheimers_prediction")
        print("✅ MLflow tracking configured")
        print("   - Tracking URI: file:./mlruns")
        print("   - Experiment: alzheimers_prediction")
        return mlflow, "mlflow"
    except Exception as e:
        print(f"❌ MLflow setup failed: {e}")
        print("🔄 Falling back to no tracking")
        return None, "none"


# Global tracker variables (will be set by setup_experiment_tracker)
tracker = None
tracker_type = "none"


def log_metrics(metrics_dict, step=None):
    """Log metrics to the appropriate tracker."""
    global tracker, tracker_type
    if tracker_type == "mlflow" and tracker:
        for key, value in metrics_dict.items():
            if step is not None:
                tracker.log_metric(key, value, step=step)
            else:
                tracker.log_metric(key, value)


def log_artifact(artifact_path, artifact_name, artifact_type="model"):
    """Log artifacts to the appropriate tracker."""
    global tracker, tracker_type
    if tracker_type == "mlflow" and tracker:
        tracker.log_artifact(str(artifact_path), artifact_name)


def log_table(dataframe, table_name):
    """Log tables to the appropriate tracker."""
    global tracker, tracker_type
    if tracker_type == "mlflow" and tracker:
        # MLflow doesn't have direct table logging, so we'll log as artifact
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            dataframe.to_csv(f.name, index=False)
            tracker.log_artifact(f.name, f"{table_name}.csv")


def log_plot(plot_data, plot_name, plot_type="roc_curve"):
    """Log plots to the appropriate tracker."""
    global tracker, tracker_type
    if tracker_type == "mlflow" and tracker:
        # For MLflow, we'll save plots as artifacts
        import matplotlib.pyplot as plt
        import tempfile
        
        if plot_type == "roc_curve":
            plt.figure(figsize=(8, 6))
            plt.plot(plot_data.get('fpr', []), plot_data.get('tpr', []))
            plt.title(f"ROC Curve - {plot_name}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
        elif plot_type == "pr_curve":
            plt.figure(figsize=(8, 6))
            plt.plot(plot_data.get('recall', []), plot_data.get('precision', []))
            plt.title(f"PR Curve - {plot_name}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plt.savefig(f.name, dpi=300, bbox_inches='tight')
            plt.close()
            tracker.log_artifact(f.name, f"{plot_name}.png")


def start_run(run_name=None, config=None):
    """Start a new run with the appropriate tracker."""
    global tracker, tracker_type
    if tracker_type == "mlflow" and tracker:
        tracker.start_run(run_name=run_name)


def end_run():
    """End the current run with the appropriate tracker."""
    global tracker, tracker_type
    if tracker_type == "mlflow" and tracker:
        tracker.end_run()


def get_run_id():
    """Get the current run ID from the appropriate tracker."""
    global tracker, tracker_type
    if tracker_type == "mlflow" and tracker:
        return tracker.active_run().info.run_id if tracker.active_run() else None
    return None
