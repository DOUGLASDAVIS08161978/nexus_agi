#!/bin/bash
# Run script for Nexus AGI
# This script activates the virtual environment and runs the main program

set -e

echo "=================================="
echo "Starting Nexus AGI"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found."
    echo "Please run ./setup.sh first to set up the environment."
    echo ""
    echo "Attempting to run without virtual environment..."
fi

# Check if nexus_agi exists
if [ ! -f "nexus_agi" ]; then
    echo "Error: nexus_agi file not found!"
    exit 1
fi

# Run the main program
echo "Running Nexus AGI..."
echo ""
python3 nexus_agi "$@"
