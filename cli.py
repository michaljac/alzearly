import typer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
def data_gen(
    n_patients: int = typer.Option(..., "--n-patients", help="Number of patients to generate"),
    years: str = typer.Option(..., "--years", help="Years to generate data for (comma-separated)"),
    rows: int = typer.Option(None, "--rows", help="Total target rows (overrides n-patients)"),
    positive_rate: float = typer.Option(0.07, "--positive-rate", help="Target positive diagnosis rate"),
    rows_per_chunk: int = typer.Option(100_000, "--rows-per-chunk", help="Rows per chunk for memory efficiency"),
    out: str = typer.Option("data/raw", "--out", help="Output directory"),
    seed: int = typer.Option(0, "--seed", help="Random seed for reproducibility"),
):
    """Generate synthetic patient-year data with realistic features"""
    from src.data_gen import generate
    
    # Parse years string to list of integers
    years_list = [int(y.strip()) for y in years.split(",")]
    
    generate(
        n_patients=n_patients,
        years=years_list,
        rows=rows,
        positive_rate=positive_rate,
        rows_per_chunk=rows_per_chunk,
        out=out,
        seed=seed,
    )

@app.command()
def preprocess(
    input_dir: str = typer.Option("data/raw", "--input-dir", help="Input directory with partitioned Parquet data"),
    output_dir: str = typer.Option("data/featurized", "--output-dir", help="Output directory for featurized data"),
    rolling_window_years: int = typer.Option(3, "--rolling-window", help="Number of years for rolling features"),
    chunk_size: int = typer.Option(100_000, "--chunk-size", help="Chunk size for processing"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
):
    """Preprocess patient-year data with rolling features using Polars Lazy"""
    from src.preprocess import preprocess
    preprocess(
        input_dir=input_dir,
        output_dir=output_dir,
        rolling_window_years=rolling_window_years,
        chunk_size=chunk_size,
        seed=seed,
    )

@app.command()
def train():
    """Train the ML model"""
    logger.info("not implemented yet")

@app.command()
def evaluate():
    """Evaluate the trained model"""
    logger.info("not implemented yet")

@app.command()
def serve_dev():
    """Run development server"""
    logger.info("not implemented yet")

if __name__ == "__main__":
    app()
