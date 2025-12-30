# AutoJudge: Programming Problem Difficulty Predictor

A machine learning system that automatically assesses programming problem difficulty using both classification and regression models. The system processes textual problem descriptions and provides dual predictions: categorical difficulty levels (Easy/Medium/Hard) and numerical difficulty scores (0-10).

![AutoJudge Demo](preview.html)

## 🎯 Key Features

- **Dual Prediction Models**: Random Forest classifier for difficulty categories and regressor for numerical scores
- **Advanced Text Analysis**: TF-IDF vectorization combined with statistical features (text length, mathematical symbols, keyword frequency)
- **Modern Web Interface**: React-based UI with real-time predictions and beautiful visualizations
- **Robust ML Pipeline**: Complete feature extraction, model training, and evaluation pipeline
- **Production-Ready API**: Flask backend with comprehensive error handling and model persistence

## 📊 Model Performance

- **Dataset**: 4,112 programming problems with balanced distribution
- **Training Set**: 3,289 problems (80%)
- **Test Set**: 823 problems (20%)
- **Classification Accuracy**: 51.6%
- **Regression MAE**: 1.90
- **Feature Dimensions**: 520 (TF-IDF + statistical features)

### Class Distribution
- **Hard**: 47.2% (1,552 problems)
- **Medium**: 34.2% (1,124 problems)  
- **Easy**: 18.6% (613 problems)

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+ with pip
- Git

### 1. Clone and Setup
```bash
git clone <repository-url>
cd autojudge-difficulty-predictor
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train the models (if not already trained)
python train_models.py

# Start the Flask API server
python app.py
```

The backend will start on `http://localhost:5001`

### 3. Frontend Setup
```bash
# In a new terminal, from project root
npm install
npm run dev
```

The frontend will start on `http://localhost:3000`

### 4. View Detailed Evaluation Metrics
Open `evaluation_dashboard.html` in your browser to see:
- **Confusion Matrix**: Detailed classification performance
- **Regression Metrics**: MAE, RMSE, R² scores
- **Class-wise Performance**: Precision, Recall, F1-score for each difficulty level
- **Dataset Statistics**: Sample distribution and feature counts

### 5. Test the System
1. Open `http://localhost:3000` in your browser
2. Enter a programming problem description
3. Click "Analyze Difficulty" to get predictions
4. View both categorical (Easy/Medium/Hard) and numerical (0-10) difficulty scores

## 🏗️ Architecture

### Frontend (React + TypeScript)
```
├── App.tsx                 # Main application component
├── components/
│   ├── Header.tsx         # Application header
│   ├── ProblemForm.tsx    # Problem input form
│   └── ResultDisplay.tsx  # Prediction results display
├── services/
│   └── mlService.ts       # ML backend API integration
└── types.ts               # TypeScript type definitions
```

### Backend (Python + Flask)
```
backend/
├── app.py                 # Flask API server
├── ml/
│   ├── data_models.py     # Data structures and validation
│   ├── dataset_loader.py  # Dataset processing utilities
│   ├── feature_extraction.py # TF-IDF and statistical features
│   └── models.py          # ML model implementations
├── models/                # Trained model artifacts
│   ├── classifier.joblib  # Random Forest classifier
│   ├── regressor.joblib   # Random Forest regressor
│   ├── feature_extractor.joblib # TF-IDF vectorizer
│   └── metadata.json     # Model performance metrics
└── data/                  # Training datasets
    ├── problems_data.jsonl # Original dataset
    ├── train_dataset.jsonl # Training split
    └── test_dataset.jsonl  # Test split
```

## 🔬 Technical Approach

### Feature Engineering
1. **Text Combination**: Merge title, description, input/output specifications
2. **Statistical Features**:
   - Text length (characters and words)
   - Mathematical symbol count
   - Algorithm keyword frequency (graph, dp, recursion, etc.)
3. **TF-IDF Vectorization**: Convert text to numerical features with 500 max features
4. **Feature Scaling**: Combine statistical and TF-IDF features

### Machine Learning Models
- **Classifier**: Random Forest with 100 estimators for Easy/Medium/Hard prediction
- **Regressor**: Random Forest with 100 estimators for numerical score (0-10)
- **Training**: 80/20 train-test split with stratified sampling
- **Evaluation**: Accuracy, precision, recall, F1-score for classification; MAE, RMSE, R² for regression

### API Endpoints
- `GET /health` - Health check and model status
- `POST /predict` - Problem difficulty prediction
- `GET /model-info` - Model performance metrics
- `GET /evaluate` - Detailed evaluation metrics with confusion matrix

## 📝 Usage Examples

### API Usage
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Two Sum",
    "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
    "input_description": "An array of integers and a single integer target.",
    "output_description": "Two indices of the numbers."
  }'
```

### Response Format
```json
{
  "difficulty_class": "Easy",
  "difficulty_score": 2.3,
  "confidence": 0.85,
  "processing_time": 0.045,
  "success": true
}
```

## 🧪 Testing

### Run Backend Tests
```bash
cd backend
pytest tests/ -v
```

### Test Coverage
- Data model validation
- Feature extraction pipeline
- Model prediction accuracy
- API endpoint functionality
- Error handling scenarios

## 🔧 Development

### Adding New Features
1. **New Statistical Features**: Add to `feature_extraction.py`
2. **Model Improvements**: Modify `models.py` and retrain
3. **UI Enhancements**: Update React components
4. **API Extensions**: Add endpoints to `app.py`

### Retraining Models
```bash
cd backend
python train_models.py
```

This will:
- Load and preprocess the dataset
- Extract features using the pipeline
- Train both classification and regression models
- Evaluate performance and save metrics
- Persist models to the `models/` directory

## 📈 Performance Metrics

### Classification Results
- **Accuracy**: 51.6%
- **Precision**: Varies by class (Easy: 0.45, Medium: 0.52, Hard: 0.54)
- **Recall**: Balanced across classes
- **F1-Score**: Weighted average of 0.51

### Regression Results
- **Mean Absolute Error**: 1.90
- **Root Mean Square Error**: 2.24
- **R² Score**: -0.043 (indicates room for improvement)

### Feature Importance
Top contributing features:
1. Text length (characters and words)
2. Mathematical symbol density
3. Algorithm-specific keywords (graph, dynamic programming)
4. TF-IDF terms related to complexity

## 🚀 Deployment

### Production Considerations
1. **Model Versioning**: Track model versions and performance
2. **API Rate Limiting**: Implement request throttling
3. **Caching**: Cache predictions for identical problems
4. **Monitoring**: Log prediction accuracy and response times
5. **Scaling**: Use gunicorn for production Flask deployment

### Docker Deployment (Optional)
```dockerfile
# Backend Dockerfile example
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset sourced from competitive programming platforms
- Built with scikit-learn, React, and Flask
- Inspired by automated difficulty assessment research

## 📞 Contact

For questions or suggestions, please open an issue or contact the development team.

---

**AutoJudge** - Making programming problem difficulty assessment objective and automated.