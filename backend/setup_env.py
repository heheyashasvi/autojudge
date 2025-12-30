#!/usr/bin/env python3
"""
Setup script for AutoJudge ML backend environment.
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        sys.exit(1)

def main():
    """Set up the Python environment and install dependencies."""
    print("Setting up AutoJudge ML Backend Environment")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("Warning: Not running in a virtual environment")
        print("Consider creating one with: python -m venv venv")
        print("And activating it with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
        
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled")
            sys.exit(0)
    
    # Install requirements
    if os.path.exists('requirements.txt'):
        run_command('pip install -r requirements.txt', 'Installing Python dependencies')
    else:
        print("requirements.txt not found, installing packages individually")
        packages = [
            'scikit-learn==1.3.2',
            'pandas==2.1.4', 
            'numpy==1.24.4',
            'flask==3.0.0',
            'flask-cors==4.0.0',
            'joblib==1.3.2',
            'hypothesis==6.92.1',
            'pytest==7.4.3'
        ]
        for package in packages:
            run_command(f'pip install {package}', f'Installing {package}')
    
    print("\n" + "=" * 50)
    print("✓ Environment setup complete!")
    print("\nTo start the Flask server:")
    print("  python app.py")
    print("\nTo run tests:")
    print("  pytest tests/")

if __name__ == '__main__':
    main()