# Technology Stack

## Frontend
- **Framework**: React 19.2.3 with TypeScript
- **Build Tool**: Vite 6.2.0
- **UI Libraries**: Recharts for data visualization
- **AI Integration**: Google Generative AI (@google/genai)

## Backend
- **Runtime**: Python 3.x with Flask 3.0.0
- **ML Libraries**: 
  - scikit-learn 1.3.2 (Random Forest models)
  - pandas 2.1.4 (data processing)
  - numpy 1.24.4 (numerical operations)
- **API**: Flask with CORS support
- **Testing**: pytest 7.4.3, Hypothesis 6.92.1 (property-based testing)

## Development Environment
- **Package Management**: npm (frontend), pip with venv (backend)
- **TypeScript Config**: ES2022 target, React JSX, path aliases with @/*
- **Environment**: .env.local for API keys

## Common Commands

### Frontend Development
```bash
npm install          # Install dependencies
npm run dev          # Start development server (port 3000)
npm run build        # Build for production
npm run preview      # Preview production build
```

### Backend Development
```bash
# Setup Python environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Or use setup script
python setup_env.py

# Run Flask server
python app.py        # Starts on port 5000

# Run tests
pytest tests/
```

### Environment Setup
1. Set `GEMINI_API_KEY` in `.env.local`
2. Activate Python virtual environment
3. Install both frontend and backend dependencies

## Architecture Notes
- **Monorepo Structure**: Frontend and backend in same repository
- **API Communication**: Frontend (port 3000) → Backend (port 5000)
- **Model Persistence**: joblib for saving/loading ML models
- **CORS Enabled**: Backend configured for cross-origin requests