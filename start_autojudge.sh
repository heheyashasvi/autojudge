#!/bin/bash

# AutoJudge Difficulty Predictor - Startup Script
echo "🚀 Starting AutoJudge Difficulty Predictor..."

# Check if Python virtual environment exists
if [ ! -d "backend/venv" ]; then
    echo "❌ Python virtual environment not found. Please run setup first:"
    echo "   cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if models exist
if [ ! -f "backend/models/classifier.joblib" ]; then
    echo "❌ ML models not found. Please train models first:"
    echo "   cd backend && source venv/bin/activate && python train_models.py"
    exit 1
fi

# Start backend
echo "📦 Starting ML Backend on port 5001..."
cd backend
source venv/bin/activate
python app.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Test backend health
echo "🔍 Testing backend health..."
if curl -s http://localhost:5001/health > /dev/null; then
    echo "✅ Backend is healthy!"
else
    echo "❌ Backend health check failed"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Check if npm is available for frontend
if command -v npm &> /dev/null; then
    echo "🌐 Starting React frontend on port 3000..."
    npm run dev &
    FRONTEND_PID=$!
    
    echo ""
    echo "🎉 AutoJudge is running!"
    echo "   Backend:  http://localhost:5001"
    echo "   Frontend: http://localhost:3000"
    echo ""
    echo "Press Ctrl+C to stop all services"
    
    # Wait for user interrupt
    trap "echo '🛑 Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT
    wait
else
    echo "⚠️  npm not found. Frontend not started."
    echo "   You can test the backend directly at: http://localhost:5001"
    echo "   Or open test_integration.html in your browser"
    echo ""
    echo "Backend is running. Press Ctrl+C to stop."
    
    # Wait for user interrupt
    trap "echo '🛑 Stopping backend...'; kill $BACKEND_PID 2>/dev/null; exit 0" INT
    wait $BACKEND_PID
fi