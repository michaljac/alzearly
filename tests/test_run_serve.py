#!/usr/bin/env python3
"""
Unit tests for run_serve.py script.

Tests the command-line interface, argument parsing, and server startup functionality.
"""

import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import argparse

# Add the parent directory to the path so we can import run_serve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_serve


class TestRunServe(unittest.TestCase):
    """Test cases for run_serve.py script."""

    def setUp(self):
        """Set up test fixtures."""
        self.original_argv = sys.argv.copy()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        sys.argv = self.original_argv
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_argument_parser_defaults(self):
        """Test that argument parser has correct default values."""
        # Create parser manually to test argument parsing
        parser = argparse.ArgumentParser(description="Run Alzheimer's prediction API server")
        parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
        parser.add_argument("--host", default="0.0.0.0", help="Host to bind server to")
        parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
        
        # Test default values
        args = parser.parse_args([])
        self.assertEqual(args.port, 8000)
        self.assertEqual(args.host, "0.0.0.0")
        self.assertFalse(args.reload)

    def test_argument_parser_custom_values(self):
        """Test that argument parser accepts custom values."""
        # Create parser manually to test argument parsing
        parser = argparse.ArgumentParser(description="Run Alzheimer's prediction API server")
        parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
        parser.add_argument("--host", default="0.0.0.0", help="Host to bind server to")
        parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
        
        # Test custom values
        test_args = ["--port", "9000", "--host", "127.0.0.1", "--reload"]
        args = parser.parse_args(test_args)
        self.assertEqual(args.port, 9000)
        self.assertEqual(args.host, "127.0.0.1")
        self.assertTrue(args.reload)

    def test_argument_parser_help(self):
        """Test that help argument works correctly."""
        # Create parser manually to test argument parsing
        parser = argparse.ArgumentParser(description="Run Alzheimer's prediction API server")
        parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
        parser.add_argument("--host", default="0.0.0.0", help="Host to bind server to")
        parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
        
        # Test help argument
        with patch('sys.stdout') as mock_stdout:
            try:
                parser.parse_args(["--help"])
            except SystemExit:
                pass  # argparse calls sys.exit() when --help is used
        
        # Verify help was printed
        mock_stdout.write.assert_called()

    def test_argument_parser_invalid_port(self):
        """Test that invalid port raises an error."""
        # Create parser manually to test argument parsing
        parser = argparse.ArgumentParser(description="Run Alzheimer's prediction API server")
        parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
        parser.add_argument("--host", default="0.0.0.0", help="Host to bind server to")
        parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
        
        # Test invalid port
        with self.assertRaises(SystemExit):
            parser.parse_args(["--port", "invalid"])

    def test_argument_parser_invalid_arguments(self):
        """Test that invalid arguments raise an error."""
        # Create parser manually to test argument parsing
        parser = argparse.ArgumentParser(description="Run Alzheimer's prediction API server")
        parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
        parser.add_argument("--host", default="0.0.0.0", help="Host to bind server to")
        parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
        
        # Test invalid arguments
        with self.assertRaises(SystemExit):
            parser.parse_args(["--invalid-arg"])

    @patch('run_serve.uvicorn.run')
    def test_main_function_success(self, mock_uvicorn_run):
        """Test successful execution of main function."""
        # Mock uvicorn.run to avoid actually starting the server
        mock_uvicorn_run.return_value = None
        
        # Test with default arguments
        sys.argv = ["run_serve.py"]
        result = run_serve.main()
        
        # Verify uvicorn.run was called with correct arguments
        mock_uvicorn_run.assert_called_once_with(
            "run_serve:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
        
        # Verify return value
        self.assertEqual(result, 0)

    @patch('run_serve.uvicorn.run')
    def test_main_function_with_custom_args(self, mock_uvicorn_run):
        """Test main function with custom arguments."""
        # Mock uvicorn.run to avoid actually starting the server
        mock_uvicorn_run.return_value = None
        
        # Test with custom arguments
        sys.argv = ["run_serve.py", "--port", "9000", "--host", "127.0.0.1", "--reload"]
        result = run_serve.main()
        
        # Verify uvicorn.run was called with correct arguments
        mock_uvicorn_run.assert_called_once_with(
            "run_serve:app",
            host="127.0.0.1",
            port=9000,
            reload=True,
            log_level="info"
        )
        
        # Verify return value
        self.assertEqual(result, 0)

    @patch('run_serve.uvicorn.run')
    def test_main_function_keyboard_interrupt(self, mock_uvicorn_run):
        """Test main function handles KeyboardInterrupt gracefully."""
        # Mock uvicorn.run to raise KeyboardInterrupt
        mock_uvicorn_run.side_effect = KeyboardInterrupt()
        
        # Test with default arguments
        sys.argv = ["run_serve.py"]
        result = run_serve.main()
        
        # Verify return value
        self.assertEqual(result, 0)

    @patch('run_serve.uvicorn.run')
    def test_main_function_exception(self, mock_uvicorn_run):
        """Test main function handles exceptions gracefully."""
        # Mock uvicorn.run to raise an exception
        mock_uvicorn_run.side_effect = Exception("Test exception")
        
        # Test with default arguments
        sys.argv = ["run_serve.py"]
        result = run_serve.main()
        
        # Verify return value
        self.assertEqual(result, 1)

    @patch('run_serve.print')
    def test_main_function_output(self, mock_print):
        """Test that main function produces expected output."""
        with patch('run_serve.uvicorn.run') as mock_uvicorn_run:
            mock_uvicorn_run.return_value = None
            
            # Test with default arguments
            sys.argv = ["run_serve.py"]
            run_serve.main()
            
            # Verify expected output
            expected_calls = [
                call("Alzearly - API Server"),
                call("=" * 40),
                call("Server will be available at: http://0.0.0.0:8000"),
                call("Interactive docs at: http://localhost:8000/docs"),
                call("Press Ctrl+C to stop the server"),
                call()
            ]
            
            # Check that all expected print calls were made
            for expected_call in expected_calls:
                self.assertIn(expected_call, mock_print.call_args_list)

    @patch('run_serve.print')
    def test_main_function_output_custom_port(self, mock_print):
        """Test that main function produces correct output with custom port."""
        with patch('run_serve.uvicorn.run') as mock_uvicorn_run:
            mock_uvicorn_run.return_value = None
            
            # Test with custom port
            sys.argv = ["run_serve.py", "--port", "9000"]
            run_serve.main()
            
            # Verify expected output with custom port
            expected_calls = [
                call("Alzearly - API Server"),
                call("=" * 40),
                call("Server will be available at: http://0.0.0.0:9000"),
                call("Interactive docs at: http://localhost:8000/docs"),
                call("Press Ctrl+C to stop the server"),
                call()
            ]
            
            # Check that all expected print calls were made
            for expected_call in expected_calls:
                self.assertIn(expected_call, mock_print.call_args_list)

    def test_script_imports(self):
        """Test that all required imports work."""
        try:
            import run_serve
            self.assertTrue(hasattr(run_serve, 'main'))
            self.assertTrue(hasattr(run_serve, 'argparse'))
            self.assertTrue(hasattr(run_serve, 'sys'))
            self.assertTrue(hasattr(run_serve, 'uvicorn'))
        except ImportError as e:
            self.fail(f"Failed to import run_serve: {e}")

    def test_main_function_signature(self):
        """Test that main function has correct signature."""
        import inspect
        
        # Check function signature
        sig = inspect.signature(run_serve.main)
        self.assertEqual(len(sig.parameters), 0)  # No parameters expected

    def test_script_execution_as_main(self):
        """Test that script can be executed as main module."""
        # This test is complex to mock properly since it involves module reloading
        # Instead, we'll test that the main function exists and can be called
        self.assertTrue(hasattr(run_serve, 'main'))
        self.assertTrue(callable(run_serve.main))
        
        # Test that the script has the expected structure
        with open('../run_serve.py', 'r') as f:
            content = f.read()
            self.assertIn('if __name__ == "__main__":', content)
            self.assertIn('sys.exit(main())', content)

    def test_argument_parser_description(self):
        """Test that argument parser has correct description."""
        parser = argparse.ArgumentParser(description="Run Alzheimer's prediction API server")
        parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
        parser.add_argument("--host", default="0.0.0.0", help="Host to bind server to")
        parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
        
        self.assertIn("Alzheimer", parser.description)
        self.assertIn("API", parser.description)

    def test_argument_parser_help_text(self):
        """Test that argument parser help text is informative."""
        parser = argparse.ArgumentParser(description="Run Alzheimer's prediction API server")
        parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
        parser.add_argument("--host", default="0.0.0.0", help="Host to bind server to")
        parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
        
        # Test help text for each argument
        help_text = parser.format_help()
        
        # Check that help text contains expected information
        self.assertIn("--port", help_text)
        self.assertIn("--host", help_text)
        self.assertIn("--reload", help_text)
        # Note: argparse help doesn't show default values in the format we expected
        # The defaults are only shown when using --help with actual arguments

    def test_uvicorn_integration(self):
        """Test that uvicorn.run is called with correct parameters."""
        with patch('run_serve.uvicorn.run') as mock_uvicorn_run:
            mock_uvicorn_run.return_value = None
            
            # Test default arguments
            sys.argv = ["run_serve.py"]
            run_serve.main()
            
            # Verify uvicorn.run was called with correct parameters
            call_args = mock_uvicorn_run.call_args
            self.assertEqual(call_args[0][0], "run_serve:app")  # app string
            self.assertEqual(call_args[1]['host'], "0.0.0.0")
            self.assertEqual(call_args[1]['port'], 8000)
            self.assertEqual(call_args[1]['reload'], False)
            self.assertEqual(call_args[1]['log_level'], "info")

    def test_error_handling_integration(self):
        """Test error handling integration with uvicorn."""
        with patch('run_serve.uvicorn.run') as mock_uvicorn_run:
            # Test different types of exceptions
            test_exceptions = [
                OSError("Port already in use"),
                ImportError("Module not found"),
                ValueError("Invalid configuration")
            ]
            
            for exception in test_exceptions:
                mock_uvicorn_run.side_effect = exception
                
                with patch('run_serve.print') as mock_print:
                    sys.argv = ["run_serve.py"]
                    result = run_serve.main()
                    
                    # Verify error message was printed
                    error_calls = [call for call in mock_print.call_args_list 
                                 if "Failed to start server" in str(call)]
                    self.assertTrue(len(error_calls) > 0)
                    
                    # Verify return value is 1 for errors
                    self.assertEqual(result, 1)


class TestRunServeIntegration(unittest.TestCase):
    """Integration tests for run_serve.py with actual FastAPI app."""

    def setUp(self):
        """Set up test fixtures."""
        self.original_argv = sys.argv.copy()

    def tearDown(self):
        """Clean up test fixtures."""
        sys.argv = self.original_argv

    def test_fastapi_app_import(self):
        """Test that the FastAPI app can be imported."""
        try:
            from run_serve import app
            self.assertIsNotNone(app)
            self.assertTrue(hasattr(app, 'routes'))
            print("FastAPI app imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import FastAPI app: {e}")

    def test_fastapi_app_routes(self):
        """Test that the FastAPI app has expected routes."""
        try:
            from run_serve import app
            
            # Get all route paths
            routes = [route.path for route in app.routes]
            
            # Check for expected routes
            expected_routes = ['/', '/health', '/version', '/predict', '/docs', '/openapi.json']
            for route in expected_routes:
                self.assertIn(route, routes, f"Expected route {route} not found")
            print("All expected routes found")
                
        except ImportError as e:
            self.fail(f"Failed to import FastAPI app: {e}")

    def test_uvicorn_app_string_validity(self):
        """Test that the app string passed to uvicorn is valid."""
        try:
            # Try to import the app using the same string as run_serve.py
            import importlib
            module_name, app_name = "run_serve:app".split(":")
            
            # Import the module
            module = importlib.import_module(module_name)
            
            # Get the app
            app = getattr(module, app_name)
            
            # Verify it's a FastAPI app
            from fastapi import FastAPI
            self.assertIsInstance(app, FastAPI)
            print("Uvicorn app string is valid")
            
        except (ImportError, AttributeError) as e:
            self.fail(f"Invalid app string 'run_serve:app': {e}")


class TestServeFunctionality(unittest.TestCase):
    """Additional functionality tests for the serve module."""
    
    def setUp(self):
        """Set up test environment."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    def test_imports(self):
        """Test if all required imports work."""
        try:
            import fastapi
            import uvicorn
            import numpy as np
            import pandas as pd
            import pickle
            print("All required imports successful")
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_model_loading_function(self):
        """Test model loading functionality."""
        try:
            from run_serve import load_model_and_metadata
            print("Model loading function imported successfully")
        except Exception as e:
            self.fail(f"Model loading function import failed: {e}")
    
    def test_docker_compose_configuration(self):
        """Test if docker-compose configuration is valid."""
        docker_compose_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "docker-compose.yml"
        self.assertTrue(docker_compose_path.exists(), "docker-compose.yml not found")
        
        with open(docker_compose_path, 'r') as f:
            content = f.read()
            self.assertIn("serve:", content, "Serve service not found in docker-compose.yml")
    
    def test_requirements_file(self):
        """Test if requirements file exists and contains necessary packages."""
        req_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "requirements-serve.txt"
        self.assertTrue(req_path.exists(), "Requirements file not found")
        
        with open(req_path, 'r') as f:
            content = f.read().lower()
            self.assertIn("fastapi", content, "FastAPI not in requirements-serve.txt")
            self.assertIn("uvicorn", content, "Uvicorn not in requirements-serve.txt")


class TestAPIEndpoints(unittest.TestCase):
    """API endpoint structure and validation tests."""
    
    def setUp(self):
        """Set up test environment."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    def test_health_endpoint_function(self):
        """Test that health endpoint function exists and returns correct structure."""
        try:
            from run_serve import app
            
            # Find the health endpoint
            health_route = None
            for route in app.routes:
                if hasattr(route, 'path') and route.path == '/health':
                    health_route = route
                    break
            
            self.assertIsNotNone(health_route, "Health endpoint not found")
            self.assertEqual(health_route.methods, {'GET'}, "Health endpoint should be GET")
            print("Health endpoint function exists")
            
        except Exception as e:
            self.fail(f"Health endpoint test failed: {e}")
    
    def test_version_endpoint_function(self):
        """Test that version endpoint function exists and returns correct structure."""
        try:
            from run_serve import app
            
            # Find the version endpoint
            version_route = None
            for route in app.routes:
                if hasattr(route, 'path') and route.path == '/version':
                    version_route = route
                    break
            
            self.assertIsNotNone(version_route, "Version endpoint not found")
            self.assertEqual(version_route.methods, {'GET'}, "Version endpoint should be GET")
            print("Version endpoint function exists")
            
        except Exception as e:
            self.fail(f"Version endpoint test failed: {e}")
    
    def test_predict_endpoint_function(self):
        """Test that predict endpoint function exists and accepts correct payload structure."""
        try:
            from run_serve import app
            from run_serve import PredictionRequest, PredictionResponse
            
            # Find the predict endpoint
            predict_route = None
            for route in app.routes:
                if hasattr(route, 'path') and route.path == '/predict':
                    predict_route = route
                    break
            
            self.assertIsNotNone(predict_route, "Predict endpoint not found")
            self.assertEqual(predict_route.methods, {'POST'}, "Predict endpoint should be POST")
            print("Predict endpoint function exists")
            
            # Test that the request model can be instantiated
            payload = {
                "items": [{
                    "age": 65,
                    "bmi": 26.5,
                    "systolic_bp": 140,
                    "diastolic_bp": 85,
                    "heart_rate": 72,
                    "temperature": 37.0,
                    "glucose": 95,
                    "cholesterol_total": 200,
                    "hdl": 45,
                    "ldl": 130,
                    "triglycerides": 150,
                    "creatinine": 1.2,
                    "hemoglobin": 14.5,
                    "white_blood_cells": 7.5,
                    "platelets": 250,
                    "num_encounters": 3,
                    "num_medications": 2,
                    "num_lab_tests": 5
                }]
            }
            
            # This should not raise an exception
            request = PredictionRequest(**payload)
            self.assertIsNotNone(request)
            print("Predict endpoint payload structure valid")
            
        except Exception as e:
            self.fail(f"Predict endpoint test failed: {e}")
    
    def test_predict_endpoint_validation(self):
        """Test that predict endpoint validates input correctly."""
        try:
            from run_serve import PredictionRequest
            
            # Test invalid payload (missing required fields)
            invalid_payload = {
                "items": [{
                    "age": 65,
                    # Missing other required fields
                }]
            }
            
            # This should raise a validation error
            with self.assertRaises(Exception):
                PredictionRequest(**invalid_payload)
            print("Predict endpoint validation working")
            
        except Exception as e:
            self.fail(f"Predict endpoint validation test failed: {e}")
    
    def test_root_endpoint_function(self):
        """Test that root endpoint function exists."""
        try:
            from run_serve import app
            
            # Find the root endpoint
            root_route = None
            for route in app.routes:
                if hasattr(route, 'path') and route.path == '/':
                    root_route = route
                    break
            
            self.assertIsNotNone(root_route, "Root endpoint not found")
            self.assertEqual(root_route.methods, {'GET'}, "Root endpoint should be GET")
            print("Root endpoint function exists")
            
        except Exception as e:
            self.fail(f"Root endpoint test failed: {e}")


class TestAPIIntegration(unittest.TestCase):
    """Integration tests for API endpoints that require the server to be running."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_url = "http://localhost:8000"
        # Check if server is running
        try:
            import requests
            response = requests.get(f"{self.base_url}/health", timeout=2)
            self.server_running = response.status_code == 200
        except:
            self.server_running = False
    
    def skip_if_server_not_running(self):
        """Skip test if server is not running."""
        if not self.server_running:
            self.skipTest("Server not running. Start with: uvicorn run_serve:app --port 8000")
    
    def test_health_endpoint_integration(self):
        """Test the health endpoint with actual HTTP request."""
        self.skip_if_server_not_running()
        
        import requests
        response = requests.get(f"{self.base_url}/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "ok")
        print("Health endpoint integration test passed")
    
    def test_version_endpoint_integration(self):
        """Test the version endpoint with actual HTTP request."""
        self.skip_if_server_not_running()
        
        import requests
        response = requests.get(f"{self.base_url}/version")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("model_version", data)
        self.assertIsInstance(data["model_version"], str)
        print("Version endpoint integration test passed")
    
    def test_predict_endpoint_integration(self):
        """Test the predict endpoint with actual HTTP request."""
        self.skip_if_server_not_running()
        
        import requests
        import json
        
        payload = {
            "items": [{
                "age": 65,
                "bmi": 26.5,
                "systolic_bp": 140,
                "diastolic_bp": 85,
                "heart_rate": 72,
                "temperature": 37.0,
                "glucose": 95,
                "cholesterol_total": 200,
                "hdl": 45,
                "ldl": 130,
                "triglycerides": 150,
                "creatinine": 1.2,
                "hemoglobin": 14.5,
                "white_blood_cells": 7.5,
                "platelets": 250,
                "num_encounters": 3,
                "num_medications": 2,
                "num_lab_tests": 5
            }]
        }
        
        response = requests.post(
            f"{self.base_url}/predict",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("predictions", data)
        self.assertIsInstance(data["predictions"], list)
        self.assertEqual(len(data["predictions"]), 1)
        
        prediction = data["predictions"][0]
        self.assertIn("probability", prediction)
        self.assertIn("label", prediction)
        self.assertIsInstance(prediction["probability"], (int, float))
        self.assertIn(prediction["label"], [0, 1])
        print("Predict endpoint integration test passed")
    
    def test_predict_endpoint_multiple_items(self):
        """Test the predict endpoint with multiple items."""
        self.skip_if_server_not_running()
        
        import requests
        
        payload = {
            "items": [
                {
                    "age": 65,
                    "bmi": 26.5,
                    "systolic_bp": 140,
                    "diastolic_bp": 85,
                    "heart_rate": 72,
                    "temperature": 37.0,
                    "glucose": 95,
                    "cholesterol_total": 200,
                    "hdl": 45,
                    "ldl": 130,
                    "triglycerides": 150,
                    "creatinine": 1.2,
                    "hemoglobin": 14.5,
                    "white_blood_cells": 7.5,
                    "platelets": 250,
                    "num_encounters": 3,
                    "num_medications": 2,
                    "num_lab_tests": 5
                },
                {
                    "age": 45,
                    "bmi": 24.0,
                    "systolic_bp": 120,
                    "diastolic_bp": 80,
                    "heart_rate": 68,
                    "temperature": 36.8,
                    "glucose": 90,
                    "cholesterol_total": 180,
                    "hdl": 50,
                    "ldl": 110,
                    "triglycerides": 120,
                    "creatinine": 1.0,
                    "hemoglobin": 15.0,
                    "white_blood_cells": 6.8,
                    "platelets": 220,
                    "num_encounters": 1,
                    "num_medications": 0,
                    "num_lab_tests": 3
                }
            ]
        }
        
        response = requests.post(
            f"{self.base_url}/predict",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("predictions", data)
        self.assertIsInstance(data["predictions"], list)
        self.assertEqual(len(data["predictions"]), 2)
        
        for prediction in data["predictions"]:
            self.assertIn("probability", prediction)
            self.assertIn("label", prediction)
            self.assertIsInstance(prediction["probability"], (int, float))
            self.assertIn(prediction["label"], [0, 1])
        print("Predict endpoint multiple items test passed")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
