import os
import json
import time
from pathlib import Path
from contextlib import contextmanager

ART_DIR = Path("artifacts/latest")
ART_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def tracker_run(run_name: str, params: dict = None):
    """Context manager for experiment tracking with pluggable backends.
    
    Args:
        run_name: Name of the training run
        params: Dictionary of parameters to log
    
    Yields:
        Dictionary with 'log' function for logging metrics
    """
    tracker = os.getenv("TRACKER", "none").lower()
    params = params or {}
    start = time.time()

    if tracker == "wandb":
        import wandb
        wandb_mode = os.getenv("WANDB_MODE", "online")
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "alz_detect"),
            name=run_name,
            mode=wandb_mode,
            config=params
        )
        try:
            yield {"log": wandb.log}
        finally:
            wandb.log({"_runtime_sec": time.time() - start})
            wandb.finish()

    elif tracker == "mlflow":
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "alz_detect"))
        with mlflow.start_run(run_name=run_name):
            for k, v in params.items():
                mlflow.log_param(k, v)
            
            def _log(d):
                for k, v in d.items():
                    mlflow.log_metric(k, float(v))
            
            try:
                yield {"log": _log}
            finally:
                pass

    else:
        # local JSON log fallback
        log_path = ART_DIR / "run_log.json"
        data = {"run_name": run_name, "params": params, "metrics": []}
        def _log(d):
            # Convert numpy values to native Python types for JSON serialization
            converted_d = {}
            for k, v in d.items():
                if hasattr(v, 'item'):  # Check if it's a numpy scalar
                    converted_d[k] = v.item()
                else:
                    converted_d[k] = v
            data["metrics"].append(converted_d)
        
        try:
            yield {"log": _log}
        finally:
            data["_runtime_sec"] = time.time() - start
            with open(log_path, "w") as f:
                json.dump(data, f, indent=2)
