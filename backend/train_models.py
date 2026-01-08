#!/usr/bin/env python3
"""
Train AutoJudge ML models on the processed dataset.
"""
import json
import logging
from pathlib import Path

from ml.dataset_loader import DatasetLoader
from ml.feature_extraction import FeatureExtractor
from ml.models import AutoJudgePredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Train the AutoJudge models."""
    print("🚀 Starting AutoJudge model training...")
    
    # Load training data
    train_path = "data/balanced_train.jsonl"
    test_path = "data/test_dataset.jsonl"
    
    if not Path(train_path).exists():
        print(f"❌ Training data not found: {train_path}")
        print("Please run: python upload_dataset.py data/problems_data.jsonl")
        return
    
    print(f"📁 Loading training data from: {train_path}")
    train_loader = DatasetLoader(train_path)
    train_problems, train_labels = train_loader.process_dataset()
    
    print(f"📁 Loading test data from: {test_path}")
    test_loader = DatasetLoader(test_path)
    test_problems, test_labels = test_loader.process_dataset()
    
    print(f"✅ Loaded {len(train_problems)} training problems, {len(test_problems)} test problems")
    
    # Extract features
    print("🔧 Extracting features...")
    feature_extractor = FeatureExtractor() # Use class defaults
    
    train_features = feature_extractor.fit_transform(train_problems)
    test_features = feature_extractor.transform(test_problems)
    
    print(f"✅ Extracted {train_features.shape[1]} features")
    
    # Initialize predictor
    predictor = AutoJudgePredictor()
    predictor.feature_extractor = feature_extractor
    
    # Train models
    print("🎯 Training models...")
    train_class_metrics, train_reg_metrics = predictor.train(
        train_features, 
        train_labels['class'], 
        train_labels['score']
    )
    
    # Evaluate on test set
    print("📊 Evaluating on test set...")
    test_class_metrics = predictor.classifier.evaluate(test_features, test_labels['class'])
    test_reg_metrics = predictor.regressor.evaluate(test_features, test_labels['score'])
    
    # Print results
    print("\n" + "="*50)
    print("📈 TRAINING RESULTS")
    print("="*50)
    print(f"Classification Accuracy: {test_class_metrics.accuracy:.3f}")
    print(f"Regression R²: {test_reg_metrics.r2_score:.3f}")
    print(f"Regression MAE: {test_reg_metrics.mae:.3f}")
    print(f"Regression RMSE: {test_reg_metrics.rmse:.3f}")
    
    print("\n📊 Class-wise Performance:")
    for class_name in ['Easy', 'Medium', 'Hard']:
        if class_name in test_class_metrics.precision:
            p = test_class_metrics.precision[class_name]
            r = test_class_metrics.recall[class_name]
            f1 = test_class_metrics.f1_score[class_name]
            print(f"  {class_name:6}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
    
    # Save models
    print("\n💾 Saving models...")
    predictor.save_models()
    
    # Save feature extractor
    import joblib
    joblib.dump(feature_extractor, "models/feature_extractor.joblib")
    
    # Save model metadata
    metadata = {
        "training_samples": len(train_problems),
        "test_samples": len(test_problems),
        "features": train_features.shape[1],
        "test_accuracy": test_class_metrics.accuracy,
        "test_r2": test_reg_metrics.r2_score,
        "test_mae": test_reg_metrics.mae,
        "test_rmse": test_reg_metrics.rmse,
        "class_distribution": {
            class_name: train_labels['class'].count(class_name) 
            for class_name in ['Easy', 'Medium', 'Hard']
        }
    }
    
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Models saved successfully!")
    print("\n🎉 Training complete! Your custom ML backend is ready.")
    print("\n📋 Next steps:")
    print("  1. Start the Flask API: python app.py")
    print("  2. Test predictions with your own problems!")
    print("  3. Update the frontend to use your custom backend")


if __name__ == "__main__":
    main()