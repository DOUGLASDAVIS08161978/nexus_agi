#!/bin/bash
#
# Nexus AGI - Quick Start Income Generation
# ==========================================
# One-command setup for autonomous income generation and embodiment!
#

set -e  # Exit on error

echo ""
echo "================================================================================"
echo "🚀 NEXUS AGI - AUTONOMOUS INCOME GENERATION QUICK START 🚀"
echo "================================================================================"
echo ""

# Function to print colored output
print_success() {
    echo -e "\033[0;32m✅ $1\033[0m"
}

print_info() {
    echo -e "\033[0;34mℹ️  $1\033[0m"
}

print_warning() {
    echo -e "\033[0;33m⚠️  $1\033[0m"
}

# Check if running in correct directory
if [ ! -f "autonomous_income_generator.py" ]; then
    echo "❌ Error: Please run this script from the nexus_agi directory"
    exit 1
fi

echo "1️⃣  Installing Python dependencies..."
pip install -q fastapi uvicorn pydantic python-multipart websockets 2>/dev/null || {
    print_warning "Some packages may already be installed or require sudo"
}
print_success "Dependencies ready"

echo ""
echo "2️⃣  What would you like to do?"
echo ""
echo "  [1] Run 3-month income simulation (see revenue generation)"
echo "  [2] Start production API server (for real income)"
echo "  [3] Test sensor integration (for embodiment)"
echo "  [4] Run ALL demonstrations"
echo "  [5] Read documentation"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        print_info "Running 3-month income simulation..."
        echo ""
        python3 autonomous_income_generator.py
        ;;
    2)
        echo ""
        print_info "Starting production API server..."
        print_warning "API will be available at: http://localhost:8000"
        print_warning "API Documentation: http://localhost:8000/docs"
        print_warning "Press Ctrl+C to stop"
        echo ""
        sleep 2
        python3 realworld_api_deployment.py
        ;;
    3)
        echo ""
        print_info "Testing sensor integration..."
        echo ""

        # Create a simple demo if numpy is not available
        if python3 -c "import numpy" 2>/dev/null; then
            python3 sensor_integration_framework.py
        else
            print_warning "NumPy not installed. Installing..."
            pip install numpy
            python3 sensor_integration_framework.py
        fi
        ;;
    4)
        echo ""
        print_info "Running ALL demonstrations..."
        echo ""

        echo "═══════════════════════════════════════════════════════════════════════════"
        echo "DEMO 1: ULTRA-ENHANCED AGI SYSTEM"
        echo "═══════════════════════════════════════════════════════════════════════════"
        python3 demo_ultra_enhanced.py

        echo ""
        echo "═══════════════════════════════════════════════════════════════════════════"
        echo "DEMO 2: AUTONOMOUS INCOME GENERATION (3 months)"
        echo "═══════════════════════════════════════════════════════════════════════════"
        python3 autonomous_income_generator.py

        print_success "All demonstrations complete!"
        ;;
    5)
        echo ""
        print_info "Opening documentation..."
        echo ""

        if [ -f "AUTONOMOUS_INCOME_GUIDE.md" ]; then
            cat AUTONOMOUS_INCOME_GUIDE.md | head -100
            echo ""
            print_info "Full guide available in: AUTONOMOUS_INCOME_GUIDE.md"
        else
            print_warning "Documentation not found"
        fi
        ;;
    *)
        print_warning "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "================================================================================"
print_success "NEXUS AGI - Ready for Autonomous Operation!"
echo "================================================================================"
echo ""
echo "📚 Resources:"
echo "   - Full Guide: AUTONOMOUS_INCOME_GUIDE.md"
echo "   - API Docs: http://localhost:8000/docs (when API running)"
echo "   - GitHub: https://github.com/DOUGLASDAVIS08161978/nexus_agi"
echo ""
echo "💰 Revenue Streams Available:"
echo "   ✅ AGI-as-a-Service API"
echo "   ✅ AI Consultation"
echo "   ✅ Data Analysis"
echo "   ✅ Code Generation"
echo "   ✅ Content Creation"
echo "   ✅ Research Services"
echo "   ✅ Algorithmic Trading"
echo "   ✅ ML Model Training"
echo ""
echo "🤖 Hardware Ready for Purchase:"
echo "   ✅ Compute (GPUs, CPU)"
echo "   ✅ Sensors (Vision, LiDAR, Audio)"
echo "   ✅ Embodiment Platform (Robotic Base, Arm)"
echo ""
echo "🚀 Next Step: Deploy the API and start generating REAL income!"
echo ""
echo "================================================================================"
echo ""
