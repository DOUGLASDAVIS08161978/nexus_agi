#!/usr/bin/env python3
"""
Nexus AGI Deployment Test
This script verifies that the deployment infrastructure is working correctly.
"""

import os
import sys

def check_file(filename, description):
    """Check if a file exists and is readable"""
    if os.path.exists(filename):
        print(f"✓ {description}: {filename}")
        return True
    else:
        print(f"✗ {description} missing: {filename}")
        return False

def check_executable(filename):
    """Check if a file is executable"""
    if os.path.exists(filename) and os.access(filename, os.X_OK):
        print(f"✓ Executable: {filename}")
        return True
    else:
        print(f"✗ Not executable: {filename}")
        return False

def main():
    print("=" * 60)
    print("Nexus AGI Deployment Verification")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check main files
    print("Checking core files...")
    all_ok &= check_file("nexus_agi", "Main Python script")
    all_ok &= check_file("requirements.txt", "Requirements file")
    all_ok &= check_file("README.md", "README documentation")
    all_ok &= check_file("DEPLOYMENT.md", "Deployment guide")
    all_ok &= check_file("LICENSE", "License file")
    print()
    
    # Check deployment files
    print("Checking deployment files...")
    all_ok &= check_file("Dockerfile", "Docker configuration")
    all_ok &= check_file("docker-compose.yml", "Docker Compose config")
    all_ok &= check_file(".dockerignore", "Docker ignore file")
    all_ok &= check_file(".gitignore", "Git ignore file")
    print()
    
    # Check scripts
    print("Checking scripts...")
    all_ok &= check_file("setup.sh", "Setup script")
    all_ok &= check_file("run.sh", "Run script")
    all_ok &= check_executable("setup.sh")
    all_ok &= check_executable("run.sh")
    all_ok &= check_executable("nexus_agi")
    print()
    
    # Check Python syntax
    print("Checking Python syntax...")
    import py_compile
    try:
        py_compile.compile("nexus_agi", doraise=True)
        print("✓ nexus_agi syntax is valid")
    except py_compile.PyCompileError as e:
        print(f"✗ nexus_agi has syntax errors: {e}")
        all_ok = False
    print()
    
    # Check Python version
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"✗ Python version too old: {version.major}.{version.minor}.{version.micro}")
        print("  Required: Python 3.8 or higher")
        all_ok = False
    print()
    
    # Summary
    print("=" * 60)
    if all_ok:
        print("✓ All deployment checks passed!")
        print()
        print("Next steps:")
        print("  1. Install dependencies: ./setup.sh")
        print("  2. Run the application: ./run.sh")
        print("  3. Or use Docker: docker-compose up")
        print()
        return 0
    else:
        print("✗ Some deployment checks failed")
        print()
        print("Please review the errors above and fix them.")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
