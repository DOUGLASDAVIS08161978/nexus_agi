# Tests directory for Nexus AGI

This directory contains test files for the Nexus AGI system.

## Running Tests

### Python Tests
```bash
pytest tests/ -v
```

### With Coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

## Test Structure

- `test_nexus_basic.py` - Basic tests for Python Nexus components
- `test_aria_basic.py` - Basic tests for JavaScript ARIA components
