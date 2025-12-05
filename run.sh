#!/bin/bash

# Room Scanner - Run Script
# This script activates the virtual environment and starts the Flask app

echo "🏠 Room Scanner - Starting..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: OPENAI_API_KEY environment variable is not set!"
    echo ""
    echo "Please set your OpenAI API key first:"
    echo ""
    echo "  export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    echo "Or add it to a .env file (see .env.example)"
    echo ""
    exit 1
fi

echo "✅ OpenAI API key is set"
echo ""

# Create necessary directories
mkdir -p uploads outputs/crops outputs/visualizations models

# Run the Flask app
python app.py
