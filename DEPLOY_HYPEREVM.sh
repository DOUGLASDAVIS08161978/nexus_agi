#!/bin/bash
# Simple wrapper to deploy TBTC to HyperEVM

export PRIVATE_KEY="0eee6f45b0af8f5a6a24744a1a978346d5bd66b41c64dc30bd18a32e246515cd"

echo "🚀 HyperEVM Deployment"
echo ""
echo "Using Python 3.12 compatible script..."
echo ""

python3 deploy_hyperevm_simple.py
