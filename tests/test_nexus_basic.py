"""
Basic tests for Nexus AGI Python components.
These tests verify the basic structure and syntax of the Python modules.
"""
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_nexus_agi_imports():
    """Test that nexus_agi.py can be imported without errors."""
    try:
        # Just check if the file can be parsed
        with open('nexus_agi.py', 'r') as f:
            content = f.read()
            compile(content, 'nexus_agi.py', 'exec')
        assert True
    except Exception as e:
        pytest.fail(f"Failed to parse nexus_agi.py: {e}")


def test_nexus_service_imports():
    """Test that nexus_service.py can be imported without errors."""
    try:
        with open('nexus_service.py', 'r') as f:
            content = f.read()
            compile(content, 'nexus_service.py', 'exec')
        assert True
    except Exception as e:
        pytest.fail(f"Failed to parse nexus_service.py: {e}")


def test_required_files_exist():
    """Test that all required files exist."""
    required_files = [
        'nexus_agi.py',
        'nexus_service.py',
        'requirements.txt',
        'README.md',
        'Dockerfile.nexus'
    ]
    
    for filename in required_files:
        assert os.path.exists(filename), f"Required file {filename} not found"


def test_requirements_format():
    """Test that requirements.txt is properly formatted."""
    with open('requirements.txt', 'r') as f:
        lines = f.readlines()
    
    # Should have at least some requirements
    non_comment_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
    assert len(non_comment_lines) > 0, "requirements.txt should contain package dependencies"


def test_dockerfile_nexus_exists():
    """Test that Dockerfile.nexus exists and contains expected content."""
    with open('Dockerfile.nexus', 'r') as f:
        content = f.read()
    
    assert 'FROM python' in content, "Dockerfile.nexus should use Python base image"
    assert 'nexus_agi.py' in content or 'nexus_service.py' in content, "Dockerfile should reference Python files"
