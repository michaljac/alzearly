#!/usr/bin/env python3
"""
Tests for the configuration system.
"""

import sys
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import load_config, DataGenConfig, ConfigLoader


class TestConfigLoading:
    """Test configuration loading functionality."""
    
    def test_load_data_gen_config(self):
        """Test loading data generation configuration."""
        config = load_config("data_gen")
        
        # Verify it's the right type
        assert isinstance(config, DataGenConfig)
        
        # Verify key parameters exist
        assert hasattr(config, 'n_patients')
        assert hasattr(config, 'years')
        assert hasattr(config, 'positive_rate')
        assert hasattr(config, 'seed')
        assert hasattr(config, 'output_dir')
        assert hasattr(config, 'target_column')
        
        # Verify parameter types
        assert isinstance(config.n_patients, int)
        assert isinstance(config.years, list)
        assert isinstance(config.positive_rate, float)
        assert isinstance(config.seed, int)
        assert isinstance(config.output_dir, str)
        assert isinstance(config.target_column, str)
        
        # Verify reasonable values
        assert config.n_patients > 0
        assert len(config.years) > 0
        assert 0.0 <= config.positive_rate <= 1.0
        assert config.seed >= 0
        
        print(f"✅ Config loaded successfully:")
        print(f"   n_patients: {config.n_patients}")
        print(f"   years: {config.years}")
        print(f"   positive_rate: {config.positive_rate}")
        print(f"   seed: {config.seed}")
        print(f"   output_dir: {config.output_dir}")
        print(f"   target_column: {config.target_column}")
    
    def test_config_loader(self):
        """Test ConfigLoader class directly."""
        loader = ConfigLoader()
        
        # Test loading data_gen config
        config = loader.load_data_gen_config()
        assert isinstance(config, DataGenConfig)
        
        # Test loading with custom filename
        config = loader.load_data_gen_config("data_gen.yaml")
        assert isinstance(config, DataGenConfig)
    
    def test_config_file_exists(self):
        """Test that config file exists."""
        config_path = Path("config/data_gen.yaml")
        assert config_path.exists(), f"Config file {config_path} does not exist"
    
    def test_yaml_structure(self):
        """Test that YAML file has expected structure."""
        loader = ConfigLoader()
        config_dict = loader.load_yaml("data_gen.yaml")
        
        # Check required sections exist
        assert "dataset" in config_dict, "Missing 'dataset' section in config"
        assert "target" in config_dict, "Missing 'target' section in config"
        assert "processing" in config_dict, "Missing 'processing' section in config"
        assert "output" in config_dict, "Missing 'output' section in config"
        
        # Check dataset section
        dataset = config_dict["dataset"]
        assert "n_patients" in dataset, "Missing 'n_patients' in dataset section"
        assert "years" in dataset, "Missing 'years' in dataset section"
        
        # Check target section
        target = config_dict["target"]
        assert "positive_rate" in target, "Missing 'positive_rate' in target section"
        assert "column_name" in target, "Missing 'column_name' in target section"
        
        # Check processing section
        processing = config_dict["processing"]
        assert "seed" in processing, "Missing 'seed' in processing section"
        assert "rows_per_chunk" in processing, "Missing 'rows_per_chunk' in processing section"
        
        # Check output section
        output = config_dict["output"]
        assert "directory" in output, "Missing 'directory' in output section"
        assert "format" in output, "Missing 'format' in output section"


class TestGetConfigScript:
    """Test the get_config.py script functionality."""
    
    def test_get_config_output_format(self):
        """Test that get_config.py outputs correct format for train.bat."""
        import subprocess
        import sys
        
        # Run get_config.py
        result = subprocess.run([sys.executable, "get_config.py"], 
                              capture_output=True, text=True)
        
        assert result.returncode == 0, f"get_config.py failed: {result.stderr}"
        
        # Parse output
        config_vars = {}
        print(f"Raw output from get_config.py:")
        print(f"STDOUT: {repr(result.stdout)}")
        print(f"STDERR: {repr(result.stderr)}")
        
        for line in result.stdout.strip().split('\n'):
            print(f"Processing line: {repr(line)}")
            if line.startswith('set CONFIG_'):
                key, value = line.split('=', 1)
                # Remove the 'set ' prefix from the key
                clean_key = key.replace('set ', '')
                config_vars[clean_key] = value
                print(f"Found config var: {clean_key} = {value}")
        
        print(f"Parsed config_vars: {config_vars}")
        
        # Verify required variables are present
        required_vars = [
            'CONFIG_N_PATIENTS',
            'CONFIG_YEARS', 
            'CONFIG_POSITIVE_RATE',
            'CONFIG_SEED',
            'CONFIG_OUTPUT_DIR',
            'CONFIG_TARGET_COLUMN'
        ]
        
        for var in required_vars:
            assert var in config_vars, f"Missing {var} in get_config.py output. Found: {list(config_vars.keys())}"
        
        # Verify values are reasonable
        assert config_vars['CONFIG_N_PATIENTS'].isdigit()
        assert len(config_vars['CONFIG_YEARS'].split(',')) > 0
        assert float(config_vars['CONFIG_POSITIVE_RATE']) > 0
        assert config_vars['CONFIG_SEED'].isdigit()
        
        print(f"✅ get_config.py output format is correct")
        print(f"   Variables found: {list(config_vars.keys())}")


if __name__ == "__main__":
    # Run tests
    print("Running configuration tests...")
    
    test_config = TestConfigLoading()
    test_config.test_config_file_exists()
    test_config.test_yaml_structure()
    test_config.test_config_loader()
    test_config.test_load_data_gen_config()
    
    test_script = TestGetConfigScript()
    test_script.test_get_config_output_format()
    
    print("✅ All configuration tests passed!")
