#!/bin/bash
# Startup script for CourseWise Chatbot
# Ensures proper environment and starts Streamlit

set -e  # Exit on error

echo "🚀 Starting CourseWise Chatbot..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "   Create it with: echo 'OPENAI_API_KEY=sk-...' > .env"
    echo ""
fi

# Check if port 8501 is in use
if lsof -ti:8501 > /dev/null 2>&1; then
    echo "⚠️  Port 8501 is already in use"
    echo "   Stopping existing Streamlit process..."
    pkill -f "streamlit run" || pkill -f "python.*streamlit" || true
    sleep 2
fi

# Check Python dependencies
echo "📦 Checking dependencies..."
python3 -c "import streamlit, chromadb, openai, dotenv" 2>/dev/null || {
    echo "❌ Missing dependencies. Installing..."
    pip3 install openai chromadb streamlit sentence-transformers python-dotenv "numpy<2.0" --quiet
}

# Start Streamlit
echo ""
echo "✅ Starting Streamlit server..."
echo "   URL: http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

cd "$(dirname "$0")"
python3 -m streamlit run streamlit_chatbot_app.py --server.port 8501

