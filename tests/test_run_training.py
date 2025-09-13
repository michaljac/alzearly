#!/usr/bin/env python3
"""
Unit tests for run_training.py

Tests the complete training pipeline orchestration including:
- Argument parsing
- Data existence checks
- Experiment tracking setup
- Pipeline execution flow
- Error handling
"""

import sys
import os
import tempfile
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import pytest

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


class TestRunTraining:
    """Test suite for run_training.py functionality."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create necessary directories
        Path("data").mkdir(exist_ok=True)
        Path("config").mkdir(exist_ok=True)
        Path("artifacts").mkdir(exist_ok=True)
        
        # Create a minimal config file
        config_content = """
model:
  input_dir: "/Data/featurized"
  output_dir: "artifacts/latest"
  max_features: 50
  test_size: 0.2
  random_state: 42
  models: ["logistic_regression"]
"""
        with open("config/model.yaml", "w") as f:
            f.write(config_content)
    
    def teardown_method(self):
        """Clean up test environment after each test."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('run_training.setup_experiment_tracker')
    @patch('run_training.generate_data')
    @patch('run_training.preprocess_data')
    @patch('run_training.train_model')
    def test_main_successful_pipeline(self, mock_train, mock_preprocess, mock_generate, mock_setup_tracker):
        """Test successful execution of the complete pipeline."""
        # Mock setup
        mock_setup_tracker.return_value = (None, "none")
        mock_generate.return_value = None
        mock_preprocess.return_value = None
        mock_train.return_value = None
        
        # Mock command line arguments
        with patch('sys.argv', ['run_training.py', '--tracker', 'none']):
            from run_training import main
            result = main()
        
        # Verify all steps were called
        assert result == 0
        mock_setup_tracker.assert_called_once()
        mock_generate.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_train.assert_called_once_with(config_file="config/model.yaml", tracker="none")
    
    @patch('run_training.setup_experiment_tracker')
    @patch('run_training.generate_data')
    @patch('run_training.preprocess_data')
    @patch('run_training.train_model')
    def test_main_skip_data_steps(self, mock_train, mock_preprocess, mock_generate, mock_setup_tracker):
        """Test pipeline with data generation and preprocessing skipped."""
        # Mock setup
        mock_setup_tracker.return_value = (None, "none")
        mock_train.return_value = None
        
        # Mock command line arguments
        with patch('sys.argv', ['run_training.py', '--tracker', 'none', '--skip-data-gen', '--skip-preprocess']):
            from run_training import main
            result = main()
        
        # Verify data steps were skipped
        assert result == 0
        mock_generate.assert_not_called()
        mock_preprocess.assert_not_called()
        mock_train.assert_called_once()
    
    @patch('run_training.setup_experiment_tracker')
    @patch('run_training.generate_data')
    @patch('run_training.preprocess_data')
    @patch('run_training.train_model')
    def test_main_existing_featurized_data(self, mock_train, mock_preprocess, mock_generate, mock_setup_tracker):
        """Test pipeline when featurized data already exists."""
        # Create featurized data directory with files
        featurized_dir = Path("/Data/featurized")
        featurized_dir.mkdir(parents=True, exist_ok=True)
        (featurized_dir / "test.parquet").touch()
        
        # Mock setup
        mock_setup_tracker.return_value = (None, "none")
        mock_train.return_value = None
        
        # Mock command line arguments
        with patch('sys.argv', ['run_training.py', '--tracker', 'none']):
            from run_training import main
            result = main()
        
        # Verify data steps were skipped due to existing data
        assert result == 0
        mock_generate.assert_not_called()
        mock_preprocess.assert_not_called()
        mock_train.assert_called_once()
    
    @patch('run_training.setup_experiment_tracker')
    @patch('run_training.generate_data')
    def test_main_data_generation_failure(self, mock_generate, mock_setup_tracker):
        """Test pipeline when data generation fails."""
        # Mock setup
        mock_setup_tracker.return_value = (None, "none")
        mock_generate.side_effect = Exception("Data generation failed")
        
        # Mock command line arguments
        with patch('sys.argv', ['run_training.py', '--tracker', 'none']):
            from run_training import main
            result = main()
        
        # Verify pipeline failed
        assert result == 1
        mock_generate.assert_called_once()
    
    @patch('run_training.setup_experiment_tracker')
    @patch('run_training.generate_data')
    @patch('run_training.preprocess_data')
    def test_main_preprocessing_failure(self, mock_preprocess, mock_generate, mock_setup_tracker):
        """Test pipeline when preprocessing fails."""
        # Mock setup
        mock_setup_tracker.return_value = (None, "none")
        mock_generate.return_value = None
        mock_preprocess.side_effect = Exception("Preprocessing failed")
        
        # Mock command line arguments
        with patch('sys.argv', ['run_training.py', '--tracker', 'none']):
            from run_training import main
            result = main()
        
        # Verify pipeline failed
        assert result == 1
        mock_generate.assert_called_once()
        mock_preprocess.assert_called_once()
    
    @patch('run_training.setup_experiment_tracker')
    @patch('run_training.generate_data')
    @patch('run_training.preprocess_data')
    @patch('run_training.train_model')
    def test_main_training_failure(self, mock_train, mock_preprocess, mock_generate, mock_setup_tracker):
        """Test pipeline when training fails."""
        # Mock setup
        mock_setup_tracker.return_value = (None, "none")
        mock_generate.return_value = None
        mock_preprocess.return_value = None
        mock_train.side_effect = Exception("Training failed")
        
        # Mock command line arguments
        with patch('sys.argv', ['run_training.py', '--tracker', 'none']):
            from run_training import main
            result = main()
        
        # Verify pipeline failed
        assert result == 1
        mock_generate.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_train.assert_called_once()
    
    @patch('run_training.setup_experiment_tracker')
    @patch('run_training.generate_data')
    @patch('run_training.preprocess_data')
    @patch('run_training.train_model')
    def test_main_different_trackers(self, mock_train, mock_preprocess, mock_generate, mock_setup_tracker):
        """Test pipeline with different experiment trackers."""
        # Mock setup
        mock_generate.return_value = None
        mock_preprocess.return_value = None
        mock_train.return_value = None
        
        trackers = ["none", "wandb", "mlflow"]
        
        for tracker in trackers:
            mock_setup_tracker.reset_mock()
            mock_train.reset_mock()
            
            # Mock command line arguments
            with patch('sys.argv', ['run_training.py', '--tracker', tracker]):
                from run_training import main
                result = main()
            
            # Verify correct tracker was used
            assert result == 0
            mock_train.assert_called_once_with(config_file="config/model.yaml", tracker=tracker)
    
    @patch('run_training.setup_experiment_tracker')
    @patch('run_training.generate_data')
    @patch('run_training.preprocess_data')
    @patch('run_training.train_model')
    def test_main_custom_config(self, mock_train, mock_preprocess, mock_generate, mock_setup_tracker):
        """Test pipeline with custom configuration file."""
        # Create custom config
        custom_config = "config/custom.yaml"
        with open(custom_config, "w") as f:
            f.write("model:\n  max_features: 100\n")
        
        # Mock setup
        mock_setup_tracker.return_value = (None, "none")
        mock_generate.return_value = None
        mock_preprocess.return_value = None
        mock_train.return_value = None
        
        # Mock command line arguments
        with patch('sys.argv', ['run_training.py', '--tracker', 'none', '--config', custom_config]):
            from run_training import main
            result = main()
        
        # Verify custom config was used
        assert result == 0
        mock_train.assert_called_once_with(config_file=custom_config, tracker="none")
    
    def test_argument_parsing(self):
        """Test argument parsing functionality."""
        from run_training import main
        
        # Test default arguments
        with patch('sys.argv', ['run_training.py']):
            with patch('run_training.setup_experiment_tracker') as mock_setup:
                with patch('run_training.generate_data') as mock_generate:
                    with patch('run_training.preprocess_data') as mock_preprocess:
                        with patch('run_training.train_model') as mock_train:
                            mock_setup.return_value = (None, "none")
                            mock_generate.return_value = None
                            mock_preprocess.return_value = None
                            mock_train.return_value = None
                            
                            result = main()
                            assert result == 0
        
        # Test all arguments
        with patch('sys.argv', ['run_training.py', '--tracker', 'wandb', '--skip-data-gen', '--skip-preprocess', '--config', 'custom.yaml']):
            with patch('run_training.setup_experiment_tracker') as mock_setup:
                with patch('run_training.generate_data') as mock_generate:
                    with patch('run_training.preprocess_data') as mock_preprocess:
                        with patch('run_training.train_model') as mock_train:
                            mock_setup.return_value = (None, "wandb")
                            mock_train.return_value = None
                            
                            result = main()
                            assert result == 0
                            mock_generate.assert_not_called()
                            mock_preprocess.assert_not_called()
                            mock_train.assert_called_once_with(config_file="custom.yaml", tracker="wandb")
    
    def test_featurized_data_detection(self):
        """Test detection of existing featurized data."""
        from run_training import main
        
        # Test with no featurized data
        with patch('sys.argv', ['run_training.py', '--tracker', 'none']):
            with patch('run_training.setup_experiment_tracker') as mock_setup:
                with patch('run_training.generate_data') as mock_generate:
                    with patch('run_training.preprocess_data') as mock_preprocess:
                        with patch('run_training.train_model') as mock_train:
                            mock_setup.return_value = (None, "none")
                            mock_generate.return_value = None
                            mock_preprocess.return_value = None
                            mock_train.return_value = None
                            
                            result = main()
                            assert result == 0
                            mock_generate.assert_called_once()
                            mock_preprocess.assert_called_once()
        
        # Test with existing featurized data
        featurized_dir = Path("/Data/featurized")
        featurized_dir.mkdir(parents=True, exist_ok=True)
        (featurized_dir / "data.parquet").touch()
        
        with patch('sys.argv', ['run_training.py', '--tracker', 'none']):
            with patch('run_training.setup_experiment_tracker') as mock_setup:
                with patch('run_training.generate_data') as mock_generate:
                    with patch('run_training.preprocess_data') as mock_preprocess:
                        with patch('run_training.train_model') as mock_train:
                            mock_setup.return_value = (None, "none")
                            mock_train.return_value = None
                            
                            result = main()
                            assert result == 0
                            mock_generate.assert_not_called()
                            mock_preprocess.assert_not_called()


class TestRunTrainingIntegration:
    """Integration tests for run_training.py with actual file system operations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create project structure
        Path("src").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        Path("config").mkdir(exist_ok=True)
        Path("artifacts").mkdir(exist_ok=True)
        
        # Create minimal source files
        self._create_minimal_src_files()
        self._create_minimal_config()
    
    def teardown_method(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_minimal_src_files(self):
        """Create minimal source files for testing."""
        # Create __init__.py files
        (Path("src") / "__init__.py").touch()
        
        # Create minimal data_gen.py
        data_gen_content = '''
def generate():
    """Mock data generation."""
    print("Mock data generation completed")
    return None
'''
        with open("src/data_gen.py", "w") as f:
            f.write(data_gen_content)
        
        # Create minimal preprocess.py
        preprocess_content = '''
def preprocess():
    """Mock preprocessing."""
    print("Mock preprocessing completed")
    return None
'''
        with open("src/preprocess.py", "w") as f:
            f.write(preprocess_content)
        
        # Create minimal train.py
        train_content = '''
def train(config_file="config/model.yaml", tracker="none"):
    """Mock training."""
    print("Mock training completed")
    return None
'''
        with open("src/train.py", "w") as f:
            f.write(train_content)
        
        # Create minimal utils.py
        utils_content = '''
def setup_experiment_tracker():
    """Mock experiment tracker setup."""
    print("🔬 Mock experiment tracking setup")
    return None, "none"
'''
        with open("utils.py", "w") as f:
            f.write(utils_content)
    
    def _create_minimal_config(self):
        """Create minimal configuration file."""
        config_content = """
model:
  input_dir: "/Data/featurized"
  output_dir: "artifacts/latest"
  max_features: 50
  test_size: 0.2
  random_state: 42
  models: ["logistic_regression"]
"""
        with open("config/model.yaml", "w") as f:
            f.write(config_content)
    
    def test_run_training_script_execution(self):
        """Test actual execution of run_training.py script."""
        # Create run_training.py
        run_training_content = '''#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_gen import generate as generate_data
from src.preprocess import preprocess as preprocess_data
from src.train import train as train_model
from utils import setup_experiment_tracker

def main():
    print("Alzheimer's Prediction Pipeline")
    print("=" * 60)
    
    # Setup experiment tracking
    tracker, tracker_type = setup_experiment_tracker()
    
    # Step 1: Data Generation
    print("\\nStep 1: Data Generation")
    print("-" * 30)
    generate_data()
    
    # Step 2: Data Preprocessing
    print("\\nStep 2: Data Preprocessing")
    print("-" * 30)
    preprocess_data()
    
    # Step 3: Model Training
    print("\\nStep 3: Model Training")
    print("-" * 30)
    train_model(tracker=tracker_type)
    
    print("\\nTraining pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
        with open("run_training.py", "w") as f:
            f.write(run_training_content)
        
        # Make executable
        os.chmod("run_training.py", 0o755)
        
        # Run the script
        try:
            result = subprocess.run(
                [sys.executable, "run_training.py"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Verify execution was successful
            assert result.returncode == 0
            assert "Alzheimer's Prediction Pipeline" in result.stdout
            assert "Mock data generation completed" in result.stdout
            assert "Mock preprocessing completed" in result.stdout
            assert "Mock training completed" in result.stdout
            assert "Training pipeline completed successfully!" in result.stdout
            
        except subprocess.TimeoutExpired:
            pytest.fail("Script execution timed out")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
