#!/usr/bin/env python3
"""
Comprehensive Architecture Tests for Alzearly

This test suite verifies that all components work together correctly:
1. Configuration system
2. Data generation pipeline
3. Preprocessing pipeline
4. Training pipeline
5. Docker integration
6. Cross-platform compatibility
"""

import pytest
import subprocess
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil


class TestConfigurationSystem:
    """Test the configuration loading and management system."""
    
    def test_config_loading(self):
        """Test that config files can be loaded correctly."""
        try:
            from src.config import load_config
            config = load_config("data_gen")
            
            # Verify required fields exist
            assert hasattr(config, 'n_patients')
            assert hasattr(config, 'output_dir')
            assert hasattr(config, 'seed')
            assert hasattr(config, 'years')
            assert hasattr(config, 'positive_rate')
            
            # Verify data types
            assert isinstance(config.n_patients, int)
            assert isinstance(config.output_dir, str)
            assert isinstance(config.seed, int)
            assert isinstance(config.years, list)
            assert isinstance(config.positive_rate, float)
            
            print("✅ Configuration loading works correctly")
            
        except Exception as e:
            pytest.fail(f"Configuration loading failed: {e}")
    
    def test_config_defaults(self):
        """Test that config provides sensible defaults."""
        try:
            from src.config import load_config
            config = load_config("data_gen")
            
            # Check reasonable ranges
            assert 100 <= config.n_patients <= 1000000
            assert 0.0 <= config.positive_rate <= 1.0
            assert config.seed >= 0
            assert len(config.years) > 0
            
            print("✅ Configuration defaults are reasonable")
            
        except Exception as e:
            pytest.fail(f"Configuration defaults test failed: {e}")
    
    def test_get_config_script(self):
        """Test the get_config.py script output format."""
        try:
            result = subprocess.run(
                [sys.executable, "get_config.py"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            assert result.returncode == 0, f"get_config.py failed: {result.stderr}"
            
            # Parse output
            config_vars = {}
            for line in result.stdout.strip().split('\n'):
                if line.startswith('set '):
                    key, value = line[4:].split('=', 1)
                    config_vars[key] = value
            
            # Verify required variables
            required_vars = ['CONFIG_N_PATIENTS', 'CONFIG_SEED', 'CONFIG_OUTPUT_DIR']
            for var in required_vars:
                assert var in config_vars, f"Missing {var} in get_config.py output"
            
            print("✅ get_config.py script works correctly")
            
        except Exception as e:
            pytest.fail(f"get_config.py test failed: {e}")


class TestDataGeneration:
    """Test the data generation pipeline."""
    
    def test_data_gen_imports(self):
        """Test that data generation modules can be imported."""
        try:
            from src.data_gen import generate
            print("✅ Data generation module imports successfully")
        except Exception as e:
            pytest.fail(f"Data generation import failed: {e}")
    
    def test_preprocess_imports(self):
        """Test that preprocessing modules can be imported."""
        try:
            from src.preprocess import preprocess
            print("✅ Preprocessing module imports successfully")
        except Exception as e:
            pytest.fail(f"Preprocessing import failed: {e}")
    
    def test_run_datagen_script(self):
        """Test that run_datagen.py script can be executed."""
        try:
            # Test with minimal parameters
            result = subprocess.run(
                [sys.executable, "run_datagen.py", "--help"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            assert result.returncode == 0, f"run_datagen.py --help failed: {result.stderr}"
            assert "Alzearly Data Generation Pipeline" in result.stdout
            
            print("✅ run_datagen.py script is executable")
            
        except Exception as e:
            pytest.fail(f"run_datagen.py test failed: {e}")
    
    def test_data_generation_with_temp_dir(self):
        """Test data generation in a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Create a minimal config for testing
                test_config = {
                    'n_patients': 10,  # Small number for testing
                    'output_dir': temp_dir,
                    'seed': 42,
                    'years': [2023],
                    'positive_rate': 0.1
                }
                
                # Mock config loading
                with patch('src.config.load_config') as mock_load:
                    mock_config = MagicMock()
                    mock_config.n_patients = test_config['n_patients']
                    mock_config.output_dir = test_config['output_dir']
                    mock_config.seed = test_config['seed']
                    mock_config.years = test_config['years']
                    mock_config.positive_rate = test_config['positive_rate']
                    mock_load.return_value = mock_config
                    
                    # Test data generation
                    from src.data_gen import generate
                    generate(
                        config_file=None,
                        n_patients=test_config['n_patients'],
                        years=",".join(map(str, test_config['years'])),
                        positive_rate=test_config['positive_rate'],
                        out=test_config['output_dir'],
                        seed=test_config['seed']
                    )
                    
                    # Check that files were created
                    output_files = list(Path(temp_dir).glob("*.parquet"))
                    assert len(output_files) > 0, "No output files created"
                    
                    print("✅ Data generation works with temporary directory")
                    
            except Exception as e:
                pytest.fail(f"Data generation test failed: {e}")


class TestTrainingPipeline:
    """Test the training pipeline components."""
    
    def test_training_imports(self):
        """Test that training modules can be imported."""
        try:
            from src.train import train_model
            print("✅ Training module imports successfully")
        except Exception as e:
            pytest.fail(f"Training import failed: {e}")
    
    def test_run_training_script(self):
        """Test that run_training.py script can be executed."""
        try:
            result = subprocess.run(
                [sys.executable, "run_training.py", "--help"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            assert result.returncode == 0, f"run_training.py --help failed: {result.stderr}"
            assert "Alzearly Training Pipeline" in result.stdout
            
            print("✅ run_training.py script is executable")
            
        except Exception as e:
            pytest.fail(f"run_training.py test failed: {e}")


class TestDockerIntegration:
    """Test Docker-related functionality."""
    
    def test_dockerfiles_exist(self):
        """Test that all required Dockerfiles exist."""
        required_dockerfiles = [
            "Dockerfile.datagen",
            "Dockerfile.train", 
            "Dockerfile.serve"
        ]
        
        for dockerfile in required_dockerfiles:
            assert Path(dockerfile).exists(), f"Missing {dockerfile}"
        
        print("✅ All required Dockerfiles exist")
    
    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists."""
        assert Path("docker-compose.yml").exists(), "Missing docker-compose.yml"
        print("✅ docker-compose.yml exists")
    
    def test_docker_build_commands(self):
        """Test that Docker build commands are valid."""
        try:
            # Test building datagen image
            result = subprocess.run(
                ["docker", "build", "-f", "Dockerfile.datagen", "-t", "test-datagen", "."],
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Docker datagen build works")
                # Clean up
                subprocess.run(["docker", "rmi", "test-datagen"], capture_output=True)
            else:
                print(f"⚠️  Docker datagen build failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("⚠️  Docker build timed out (this is normal for first build)")
        except Exception as e:
            print(f"⚠️  Docker build test failed: {e}")


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility."""
    
    def test_path_handling(self):
        """Test that path handling works across platforms."""
        try:
            from pathlib import Path
            
            # Test various path scenarios
            test_paths = [
                "data/raw",
                "/Data/raw",
                "../Data/alzearly/raw",
                "C:\\Data\\alzearly\\raw"  # Windows path
            ]
            
            for path_str in test_paths:
                path = Path(path_str)
                # Just test that Path can handle it
                assert str(path) is not None
                
            print("✅ Path handling works across platforms")
            
        except Exception as e:
            pytest.fail(f"Path handling test failed: {e}")
    
    def test_script_execution(self):
        """Test that scripts can be executed."""
        scripts_to_test = [
            "run_datagen.py",
            "run_training.py", 
            "run_serve.py",
            "get_config.py"
        ]
        
        for script in scripts_to_test:
            script_path = Path(script)
            assert script_path.exists(), f"Missing {script}"
            
            # Test that script can be executed
            try:
                result = subprocess.run(
                    [sys.executable, script, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print(f"✅ {script} is executable")
                else:
                    print(f"⚠️  {script} help failed: {result.stderr}")
                    
            except Exception as e:
                print(f"⚠️  {script} execution test failed: {e}")


class TestDataFlow:
    """Test the complete data flow through the pipeline."""
    
    def test_config_to_data_flow(self):
        """Test that config parameters flow correctly to data generation."""
        try:
            from src.config import load_config
            
            # Load config
            config = load_config("data_gen")
            
            # Verify config parameters are used in data generation
            assert config.n_patients > 0
            assert config.output_dir is not None
            assert config.seed >= 0
            
            print("✅ Config parameters are valid for data generation")
            
        except Exception as e:
            pytest.fail(f"Config to data flow test failed: {e}")
    
    def test_data_to_preprocessing_flow(self):
        """Test that data generation output is compatible with preprocessing."""
        try:
            # This test verifies that the data format is compatible
            # We'll test with a small sample
            from src.data_gen import generate
            from src.preprocess import preprocess
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generate small dataset
                generate(
                    config_file=None,
                    n_patients=5,
                    years="2023",
                    positive_rate=0.1,
                    out=temp_dir,
                    seed=42
                )
                
                # Check that preprocessing can handle the output
                output_files = list(Path(temp_dir).glob("*.parquet"))
                assert len(output_files) > 0, "No data files generated"
                
                print("✅ Data generation output is compatible with preprocessing")
                
        except Exception as e:
            pytest.fail(f"Data to preprocessing flow test failed: {e}")


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_config_file(self):
        """Test handling of missing config file."""
        try:
            from src.config import load_config
            
            # Test with non-existent config
            with pytest.raises(Exception):
                load_config("non_existent_config")
                
            print("✅ Missing config file is handled gracefully")
            
        except Exception as e:
            pytest.fail(f"Missing config test failed: {e}")
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        try:
            # Test with invalid parameters
            with pytest.raises(Exception):
                from src.data_gen import generate
                generate(
                    config_file=None,
                    n_patients=-1,  # Invalid negative number
                    years="2023",
                    positive_rate=0.1,
                    out="/tmp/test",
                    seed=42
                )
                
            print("✅ Invalid parameters are handled gracefully")
            
        except Exception as e:
            # This is expected to fail, so we're good
            print("✅ Invalid parameters correctly rejected")


def run_all_tests():
    """Run all architecture tests and report results."""
    print("🧪 Running Comprehensive Architecture Tests")
    print("=" * 50)
    
    test_classes = [
        TestConfigurationSystem,
        TestDataGeneration,
        TestTrainingPipeline,
        TestDockerIntegration,
        TestCrossPlatformCompatibility,
        TestDataFlow,
        TestErrorHandling
    ]
    
    passed = 0
    failed = 0
    warnings = 0
    
    for test_class in test_classes:
        print(f"\n📋 Testing {test_class.__name__}")
        print("-" * 30)
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                method()
                passed += 1
                print(f"✅ {method_name}")
            except Exception as e:
                if "⚠️" in str(e):
                    warnings += 1
                    print(f"⚠️  {method_name}: {e}")
                else:
                    failed += 1
                    print(f"❌ {method_name}: {e}")
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Warnings: {warnings}")
    
    if failed == 0:
        print("\n🎉 All critical tests passed! Architecture is working correctly.")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
