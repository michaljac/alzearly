"""
Utility functions for the Alzearly pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
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
        'wandb': 'wandb',
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
    # Check if we're in non-interactive mode (Docker container)
    import os
    if os.environ.get('NON_INTERACTIVE', 'false').lower() == 'true':
        print("\n🔬 Experiment Tracking Setup (Non-interactive mode)")
        print("=" * 50)
        print("✅ Using no experiment tracking for Docker container")
        return None, "none"
    
    print("\n🔬 Experiment Tracking Setup")
    print("=" * 50)
    
    # Present menu to user
    print("Select experiment tracker:")
    print("1. Weights & Biases (wandb)")
    print("2. MLflow (local)")
    print("3. No tracking")
    
    while True:
        try:
            choice = input("\nEnter choice (1-3, default=1): ").strip()
            if choice == "":
                choice = "1"
            
            if choice in ["1", "2", "3"]:
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Setup interrupted. Using no tracking.")
            return None, "none"
    
    # Handle wandb selection
    if choice == "1":
        return setup_wandb()
    
    # Handle MLflow selection
    elif choice == "2":
        return setup_mlflow()
    
    # Handle no tracking
    else:
        print("✅ No experiment tracking will be used.")
        return None, "none"


def setup_wandb():
    """Set up Weights & Biases tracking."""
    print("\n🔧 Setting up Weights & Biases...")
    
    # Try to import wandb
    try:
        import wandb
        print("✅ wandb available")
    except ImportError:
        print("❌ wandb not available - please install it in the Dockerfile")
        print("🔄 Falling back to no tracking")
        return None, "none"
    
    # Check for API key in environment
    import os
    api_key = os.environ.get('WANDB_API_KEY')
    
    if api_key:
        try:
            wandb.login(key=api_key)
            print("✅ Wandb login successful")
            return wandb, "wandb"
        except Exception as e:
            print(f"❌ Wandb login failed: {e}")
            print("🔄 Falling back to interactive setup")
    
    # No API key in environment - ask user
    print("\n📝 Weights & Biases Setup:")
    print("   - Get your API key from: https://wandb.ai/settings")
    print("   - Or press Enter to run in disabled mode")
    
    try:
        user_key = input("Enter API key (or press Enter for disabled mode): ").strip()
        
        if user_key:
            try:
                wandb.login(key=user_key)
                print("✅ Wandb login successful")
                return wandb, "wandb"
            except Exception as e:
                print(f"❌ Wandb login failed: {e}")
                print("🔄 Falling back to disabled mode")
        
        # User chose disabled mode or login failed
        print("🔄 Running wandb in disabled mode...")
        os.environ["WANDB_MODE"] = "disabled"
        try:
            wandb.init(mode="disabled")
            print("✅ Wandb disabled mode initialized")
            return wandb, "wandb"
        except Exception as e:
            print(f"❌ Wandb disabled mode failed: {e}")
            print("🔄 Falling back to no tracking")
            return None, "none"
            
    except (KeyboardInterrupt, EOFError):
        print("\n🔄 User cancelled - running in disabled mode...")
        os.environ["WANDB_MODE"] = "disabled"
        try:
            wandb.init(mode="disabled")
            print("✅ Wandb disabled mode initialized")
            return wandb, "wandb"
        except Exception as e:
            print(f"❌ Wandb disabled mode failed: {e}")
            print("🔄 Falling back to no tracking")
            return None, "none"


def setup_mlflow():
    """Set up MLflow tracking."""
    print("\n🔧 Setting up MLflow...")
    
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
    if tracker_type == "wandb" and tracker:
        if step is not None:
            metrics_dict["step"] = step
        tracker.log(metrics_dict)
    elif tracker_type == "mlflow" and tracker:
        for key, value in metrics_dict.items():
            if step is not None:
                tracker.log_metric(key, value, step=step)
            else:
                tracker.log_metric(key, value)


def log_artifact(artifact_path, artifact_name, artifact_type="model"):
    """Log artifacts to the appropriate tracker."""
    global tracker, tracker_type
    if tracker_type == "wandb" and tracker:
        artifact = tracker.Artifact(name=artifact_name, type=artifact_type)
        artifact.add_dir(str(artifact_path))
        tracker.log_artifact(artifact)
    elif tracker_type == "mlflow" and tracker:
        tracker.log_artifact(str(artifact_path), artifact_name)


def log_table(dataframe, table_name):
    """Log tables to the appropriate tracker."""
    global tracker, tracker_type
    if tracker_type == "wandb" and tracker:
        tracker.log({table_name: tracker.Table(dataframe=dataframe)})
    elif tracker_type == "mlflow" and tracker:
        # MLflow doesn't have direct table logging, so we'll log as artifact
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            dataframe.to_csv(f.name, index=False)
            tracker.log_artifact(f.name, f"{table_name}.csv")


def log_plot(plot_data, plot_name, plot_type="roc_curve"):
    """Log plots to the appropriate tracker."""
    global tracker, tracker_type
    if tracker_type == "wandb" and tracker:
        if plot_type == "roc_curve":
            tracker.log({plot_name: tracker.plot.roc_curve(**plot_data)})
        elif plot_type == "pr_curve":
            tracker.log({plot_name: tracker.plot.pr_curve(**plot_data)})
        elif plot_type == "confusion_matrix":
            tracker.log({plot_name: tracker.plot.confusion_matrix(**plot_data)})
    elif tracker_type == "mlflow" and tracker:
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
    if tracker_type == "wandb" and tracker:
        if config is None:
            config = {}
        # Set up meaningful project and entity names
        project_name = "alzheimers-detection"
        entity_name = None  # Use default entity (user's account)
        
        tracker.init(
            project=project_name,
            entity=entity_name,
            name=run_name,
            config=config,
            tags=["alzheimers", "ml", "healthcare"]
        )
    elif tracker_type == "mlflow" and tracker:
        tracker.start_run(run_name=run_name)


def end_run():
    """End the current run with the appropriate tracker."""
    global tracker, tracker_type
    if tracker_type == "wandb" and tracker:
        tracker.finish()
    elif tracker_type == "mlflow" and tracker:
        tracker.end_run()


def get_run_id():
    """Get the current run ID from the appropriate tracker."""
    global tracker, tracker_type
    if tracker_type == "wandb" and tracker and tracker.run:
        return tracker.run.id
    elif tracker_type == "mlflow" and tracker:
        return tracker.active_run().info.run_id if tracker.active_run() else None
    return None
