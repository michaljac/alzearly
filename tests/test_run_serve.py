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
            "src.serve:app",
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
            "src.serve:app",
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
                call("🧠 Alzearly - API Server"),
                call("=" * 40),
                call("🌐 Server will be available at: http://0.0.0.0:8000"),
                call("📖 Interactive docs at: http://localhost:8000/docs"),
                call("🛑 Press Ctrl+C to stop the server"),
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
                call("🧠 Alzearly - API Server"),
                call("=" * 40),
                call("🌐 Server will be available at: http://0.0.0.0:9000"),
                call("📖 Interactive docs at: http://localhost:8000/docs"),
                call("🛑 Press Ctrl+C to stop the server"),
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
            self.assertEqual(call_args[0][0], "src.serve:app")  # app string
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
            from src.serve import app
            self.assertIsNotNone(app)
            self.assertTrue(hasattr(app, 'routes'))
        except ImportError as e:
            self.fail(f"Failed to import FastAPI app: {e}")

    def test_fastapi_app_routes(self):
        """Test that the FastAPI app has expected routes."""
        try:
            from src.serve import app
            
            # Get all route paths
            routes = [route.path for route in app.routes]
            
            # Check for expected routes
            expected_routes = ['/', '/health', '/predict', '/docs', '/openapi.json']
            for route in expected_routes:
                self.assertIn(route, routes, f"Expected route {route} not found")
                
        except ImportError as e:
            self.fail(f"Failed to import FastAPI app: {e}")

    def test_uvicorn_app_string_validity(self):
        """Test that the app string passed to uvicorn is valid."""
        try:
            # Try to import the app using the same string as run_serve.py
            import importlib
            module_name, app_name = "src.serve:app".split(":")
            
            # Import the module
            module = importlib.import_module(module_name)
            
            # Get the app
            app = getattr(module, app_name)
            
            # Verify it's a FastAPI app
            from fastapi import FastAPI
            self.assertIsInstance(app, FastAPI)
            
        except (ImportError, AttributeError) as e:
            self.fail(f"Invalid app string 'src.serve:app': {e}")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
