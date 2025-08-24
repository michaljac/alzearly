import typer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
def data_gen(
    config_file: str = typer.Option("config/data_gen.yaml", "--config", help="Configuration file path"),
    n_patients: int = typer.Option(None, "--n-patients", help="Override number of patients from config"),
    years: str = typer.Option(None, "--years", help="Override years from config (comma-separated)"),
    positive_rate: float = typer.Option(None, "--positive-rate", help="Override positive rate from config"),
    out: str = typer.Option(None, "--out", help="Override output directory from config"),
    seed: int = typer.Option(None, "--seed", help="Override seed from config"),
):
    """Generate synthetic patient-year data with realistic features"""
    from src.data_gen import generate
    
    generate(
        config_file=config_file,
        n_patients=n_patients,
        years=years,
        positive_rate=positive_rate,
        out=out,
        seed=seed,
    )

@app.command()
def preprocess(
    config_file: str = typer.Option("config/preprocess.yaml", "--config", help="Configuration file path"),
    input_dir: str = typer.Option(None, "--input-dir", help="Override input directory from config"),
    output_dir: str = typer.Option(None, "--output-dir", help="Override output directory from config"),
    rolling_window_years: int = typer.Option(None, "--rolling-window", help="Override rolling window from config"),
    chunk_size: int = typer.Option(None, "--chunk-size", help="Override chunk size from config"),
    seed: int = typer.Option(None, "--seed", help="Override seed from config"),
):
    """Preprocess patient-year data with rolling features using Polars Lazy."""
    from src.preprocess import preprocess as preprocess_data
    
    preprocess_data(
        config_file=config_file,
        input_dir=input_dir,
        output_dir=output_dir,
        rolling_window_years=rolling_window_years,
        chunk_size=chunk_size,
        seed=seed,
    )

@app.command()
def train(
    config_file: str = typer.Option("config/model.yaml", "--config", help="Configuration file path"),
    input_dir: str = typer.Option(None, "--input-dir", "--in", help="Override input directory from config"),
    output_dir: str = typer.Option(None, "--output-dir", "--artifacts", help="Override output directory from config"),
    max_features: int = typer.Option(None, "--max-features", help="Override max features from config"),
    handle_imbalance: str = typer.Option(None, "--handle-imbalance", help="Override imbalance handling from config"),
    wandb_project: str = typer.Option(None, "--wandb-project", help="Override wandb project from config"),
    wandb_entity: str = typer.Option(None, "--wandb-entity", help="Override wandb entity from config"),
):
    """Train machine learning models for Alzheimer's prediction"""
    from src.train import train as train_models
    
    train_models(
        config_file=config_file,
        input_dir=input_dir,
        output_dir=output_dir,
        max_features=max_features,
        handle_imbalance=handle_imbalance,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )

@app.command()
def evaluate(
    model_path: str = typer.Argument(..., help="Path to trained model (.pkl)"),
    data_path: str = typer.Argument(..., help="Path to evaluation data"),
    output_dir: str = typer.Option("artifacts", "--output", help="Output directory for results"),
):
    """Evaluate a trained model with comprehensive metrics"""
    from src.evaluate import evaluate_model
    
    evaluate_model(model_path=model_path, data_path=data_path, output_dir=output_dir)

@app.command()
def serve_dev():
    """Run development server"""
    from src.serve import app
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    app()
