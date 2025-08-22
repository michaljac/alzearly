import typer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
def data_gen():
    """Generate synthetic data for the project"""
    logger.info("not implemented yet")

@app.command()
def preprocess():
    """Preprocess the data"""
    logger.info("not implemented yet")

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
