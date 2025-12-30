# AutoJudge ML Backend

Machine learning backend for the AutoJudge difficulty predictor system.

## Setup

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or run the setup script:
```bash
python setup_env.py
```

## Running the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Predict problem difficulty

## Testing

```bash
pytest tests/
```

## Project Structure

```
backend/
├── app.py              # Flask application
├── requirements.txt    # Python dependencies
├── setup_env.py       # Environment setup script
├── ml/                 # ML components
│   ├── __init__.py
│   ├── feature_extraction.py
│   ├── models.py
│   └── evaluation.py
├── tests/              # Test files
│   ├── __init__.py
│   └── test_*.py
└── models/             # Saved model files
```