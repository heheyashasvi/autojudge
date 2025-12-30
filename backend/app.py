"""
Flask API server for AutoJudge ML model endpoints.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import joblib
import numpy as np
from pathlib import Path
import time

from ml.data_models import ProblemText, create_problem_from_dict
from ml.models import AutoJudgePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables for models
predictor = None
feature_extractor = None
model_loaded = False

def load_models():
    """Load trained models on startup."""
    global predictor, feature_extractor, model_loaded
    
    try:
        models_dir = Path("models")
        if not models_dir.exists():
            logger.error("Models directory not found. Please train models first.")
            return False
        
        # Load predictor
        predictor = AutoJudgePredictor()
        if not predictor.load_models():
            logger.error("Failed to load ML models")
            return False
        
        # Load feature extractor
        feature_extractor_path = models_dir / "feature_extractor.joblib"
        if feature_extractor_path.exists():
            feature_extractor = joblib.load(feature_extractor_path)
            logger.info("Feature extractor loaded successfully")
        else:
            logger.error("Feature extractor not found")
            return False
        
        model_loaded = True
        logger.info("✅ All models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy", 
        "service": "autojudge-ml-backend",
        "models_loaded": model_loaded
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict difficulty for a programming problem using trained ML models.
    
    Expected JSON payload:
    {
        "title": "Problem title",
        "description": "Problem description",
        "input_description": "Input format description", 
        "output_description": "Output format description"
    }
    
    Returns:
    {
        "difficulty_class": "Easy|Medium|Hard",
        "difficulty_score": float,
        "confidence": float,
        "processing_time": float,
        "success": true
    }
    """
    start_time = time.time()
    
    try:
        if not model_loaded:
            return jsonify({
                "error": "ML models not loaded. Please check server logs.",
                "success": False
            }), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "success": False
            }), 400
            
        # Validate required fields
        required_fields = ['title', 'description', 'input_description', 'output_description']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {missing_fields}",
                "success": False
            }), 400
        
        # Create problem text object
        try:
            problem = create_problem_from_dict(data)
        except Exception as e:
            return jsonify({
                "error": f"Invalid problem data: {str(e)}",
                "success": False
            }), 400
        
        # Extract features
        try:
            features = feature_extractor.transform([problem])
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return jsonify({
                "error": "Feature extraction failed",
                "success": False
            }), 500
        
        # Make prediction
        try:
            predictions = predictor.predict(features)
            result = predictions[0]  # Get first (and only) prediction
            
            processing_time = time.time() - start_time
            
            return jsonify({
                "difficulty_class": result.difficulty_class,
                "difficulty_score": round(result.difficulty_score, 2),
                "confidence": round(result.confidence, 3) if result.confidence else None,
                "processing_time": round(processing_time, 3),
                "success": True
            })
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({
                "error": "Prediction failed",
                "success": False
            }), 500
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "success": False
        }), 500

@app.route('/evaluate', methods=['GET'])
def evaluate_models():
    """
    Get detailed evaluation metrics including confusion matrix and regression metrics.
    """
    if not model_loaded:
        return jsonify({
            "error": "Models not loaded",
            "success": False
        }), 503
    
    try:
        # Load test data for evaluation
        from ml.dataset_loader import DatasetLoader
        
        test_path = Path("data/test_dataset.jsonl")
        if not test_path.exists():
            return jsonify({
                "error": "Test dataset not found",
                "success": False
            }), 404
        
        # Load and process test data
        test_loader = DatasetLoader(str(test_path))
        test_problems, test_labels = test_loader.process_dataset()
        
        # Extract features
        test_features = feature_extractor.transform(test_problems)
        
        # Evaluate classification model
        class_metrics = predictor.classifier.evaluate(test_features, test_labels['class'])
        
        # Evaluate regression model  
        reg_metrics = predictor.regressor.evaluate(test_features, test_labels['score'])
        
        # Format confusion matrix for JSON
        confusion_matrix_data = {
            "labels": ["Easy", "Medium", "Hard"],
            "matrix": class_metrics.confusion_matrix.tolist()
        }
        
        # Calculate additional metrics
        total_samples = len(test_labels['class'])
        class_distribution = {
            class_name: test_labels['class'].count(class_name) 
            for class_name in ['Easy', 'Medium', 'Hard']
        }
        
        return jsonify({
            "evaluation_results": {
                "classification": {
                    "accuracy": round(class_metrics.accuracy, 4),
                    "precision": {k: round(v, 4) for k, v in class_metrics.precision.items()},
                    "recall": {k: round(v, 4) for k, v in class_metrics.recall.items()},
                    "f1_score": {k: round(v, 4) for k, v in class_metrics.f1_score.items()},
                    "confusion_matrix": confusion_matrix_data
                },
                "regression": {
                    "mae": round(reg_metrics.mae, 4),
                    "rmse": round(reg_metrics.rmse, 4),
                    "r2_score": round(reg_metrics.r2_score, 4)
                },
                "dataset_info": {
                    "test_samples": total_samples,
                    "class_distribution": class_distribution,
                    "features": test_features.shape[1]
                }
            },
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error in evaluation endpoint: {e}")
        return jsonify({
            "error": f"Evaluation failed: {str(e)}",
            "success": False
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about loaded models."""
    if not model_loaded:
        return jsonify({
            "error": "Models not loaded",
            "success": False
        }), 503
    
    try:
        # Load metadata
        metadata_path = Path("models/metadata.json")
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {"message": "No metadata available"}
        
        return jsonify({
            "model_info": metadata,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            "error": "Failed to get model info",
            "success": False
        }), 500

if __name__ == '__main__':
    print("🚀 Starting AutoJudge ML Backend...")
    print("📦 Loading trained models...")
    
    if load_models():
        print("✅ Models loaded successfully!")
        print("🌐 Starting Flask server on http://localhost:5001")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("❌ Failed to load models. Please train models first:")
        print("   python train_models.py")
        exit(1)