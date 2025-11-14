"""
Basic tests for ARIA JavaScript components.
These tests verify the basic structure and syntax of the JavaScript modules.
"""
import pytest
import subprocess
import os


def test_aria_js_syntax():
    """Test that aria.js has valid JavaScript syntax."""
    result = subprocess.run(
        ['node', '--check', 'aria.js'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"aria.js syntax check failed: {result.stderr}"


def test_aria_service_syntax():
    """Test that aria_service.js has valid JavaScript syntax."""
    result = subprocess.run(
        ['node', '--check', 'aria_service.js'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"aria_service.js syntax check failed: {result.stderr}"


def test_package_json_exists():
    """Test that package.json exists and is valid JSON."""
    import json
    
    assert os.path.exists('package.json'), "package.json should exist"
    
    with open('package.json', 'r') as f:
        data = json.load(f)
    
    assert 'name' in data, "package.json should have a name field"
    assert 'version' in data, "package.json should have a version field"


def test_dockerfile_aria_exists():
    """Test that Dockerfile.aria exists and contains expected content."""
    with open('Dockerfile.aria', 'r') as f:
        content = f.read()
    
    assert 'FROM node' in content, "Dockerfile.aria should use Node base image"
    assert 'aria' in content.lower(), "Dockerfile should reference ARIA files"
