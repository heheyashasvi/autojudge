#!/usr/bin/env python3
"""
Train improved AutoJudge ML models with enhanced accuracy.
"""
import json
import logging
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import time

from ml.dataset_loader import DatasetLoader
from ml.improved_feature_extraction import ImprovedFeatureExtractor
from ml.improved_models import ImprovedAutoJudgePredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Train the improved AutoJudge models with reduced dataset for faster training."""
    print("🚀 Starting AutoJudge IMPROVED model training (FAST MODE)...")
    print("🎯 Target: Achieve >60% classification accuracy with reduced dataset")
    
    start_time = time.time()
    
    # Load training data
    train_path = "data/train_dataset.jsonl"
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
    
    # REDUCE DATASET SIZE FOR FASTER TRAINING
    print("⚡ Reducing dataset size for faster training...")
    
    # Use only 30% of training data (stratified sampling)
    train_class_labels = [label for label in train_labels['class']]
    train_score_labels = [label for label in train_labels['score']]
    
    train_problems_reduced, _, train_class_reduced, _, train_score_reduced, _ = train_test_split(
        train_problems, train_class_labels, train_score_labels,
        test_size=0.7, random_state=42, stratify=train_class_labels
    )
    
    # Use only 50% of test data
    test_class_labels = [label for label in test_labels['class']]
    test_score_labels = [label for label in test_labels['score']]
    
    test_problems_reduced, _, test_class_reduced, _, test_score_reduced, _ = train_test_split(
        test_problems, test_class_labels, test_score_labels,
        test_size=0.5, random_state=42, stratify=test_class_labels
    )
    
    print(f"📊 Reduced dataset: {len(train_problems_reduced)} train, {len(test_problems_reduced)} test")
    
    # Create validation split from reduced training data
    train_problems_split, val_problems, train_class_split, val_class_labels = train_test_split(
        train_problems_reduced, train_class_reduced, test_size=0.2, random_state=42, 
        stratify=train_class_reduced
    )
    
    train_score_split, val_score_labels = train_test_split(
        train_score_reduced, test_size=0.2, random_state=42
    )
    
    print(f"📊 Final split: {len(train_problems_split)} train, {len(val_problems)} validation, {len(test_problems_reduced)} test")
    
    # Extract features with improved extractor (reduced features for speed)
    print("🔧 Extracting enhanced features...")
    feature_extractor = ImprovedFeatureExtractor(max_features=1000)  # Reduced from 2000
    
    train_features = feature_extractor.fit_transform(train_problems_split)
    val_features = feature_extractor.transform(val_problems)
    test_features = feature_extractor.transform(test_problems_reduced)
    
    print(f"✅ Extracted {train_features.shape[1]} enhanced features")
    
    # Try only the best algorithm for speed (skip SVR which is slow)
    algorithms = ['gradient_boosting']  # Only test the best one
    best_accuracy = 0
    best_algorithm = None
    best_predictor = None
    
    for algorithm in algorithms:
        print(f"\n🧪 Testing {algorithm} regressor...")
        
        # Initialize improved predictor with reduced hyperparameter tuning
        predictor = ImprovedAutoJudgePredictor(regressor_algorithm=algorithm)
        predictor.feature_extractor = feature_extractor
        
        # Disable hyperparameter tuning for speed
        predictor.classifier.tune_hyperparameters = False
        predictor.regressor.tune_hyperparameters = False
        
        # Train models
        print(f"🎯 Training improved models with {algorithm}...")
        train_class_metrics, train_reg_metrics = predictor.train(
            train_features, 
            train_class_split, 
            train_score_split
        )
        
        # Evaluate on validation set
        print("📊 Evaluating on validation set...")
        val_class_metrics = predictor.classifier.evaluate(val_features, val_class_labels)
        val_reg_metrics = predictor.regressor.evaluate(val_features, val_score_labels)
        
        print(f"📈 {algorithm.upper()} VALIDATION RESULTS:")
        print(f"   Classification Accuracy: {val_class_metrics.accuracy:.3f}")
        print(f"   Regression R²: {val_reg_metrics.r2_score:.3f}")
        print(f"   Regression MAE: {val_reg_metrics.mae:.3f}")
        
        # Track best model
        if val_class_metrics.accuracy > best_accuracy:
            best_accuracy = val_class_metrics.accuracy
            best_algorithm = algorithm
            best_predictor = predictor
    
    print(f"\n🏆 Best algorithm: {best_algorithm} with {best_accuracy:.3f} accuracy")
    
    # Final evaluation on test set with best model
    print("\n📊 Final evaluation on test set...")
    test_class_metrics = best_predictor.classifier.evaluate(test_features, test_class_reduced)
    test_reg_metrics = best_predictor.regressor.evaluate(test_features, test_score_reduced)
    
    # Print detailed results
    print("\n" + "="*60)
    print("📈 IMPROVED MODEL RESULTS")
    print("="*60)
    print(f"🎯 Classification Accuracy: {test_class_metrics.accuracy:.3f}")
    print(f"📊 Regression R²: {test_reg_metrics.r2_score:.3f}")
    print(f"📉 Regression MAE: {test_reg_metrics.mae:.3f}")
    print(f"📉 Regression RMSE: {test_reg_metrics.rmse:.3f}")
    print(f"⚡ Best Algorithm: {best_algorithm}")
    
    # Improvement comparison
    original_accuracy = 0.5164  # From metadata.json
    improvement = (test_class_metrics.accuracy - original_accuracy) * 100
    print(f"🚀 Accuracy Improvement: +{improvement:.1f} percentage points")
    
    print("\n📊 Class-wise Performance:")
    for class_name in ['Easy', 'Medium', 'Hard']:
        if class_name in test_class_metrics.precision:
            p = test_class_metrics.precision[class_name]
            r = test_class_metrics.recall[class_name]
            f1 = test_class_metrics.f1_score[class_name]
            print(f"  {class_name:6}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
    
    # Test prediction alignment (reduced sample)
    print("\n🔄 Testing prediction alignment...")
    sample_predictions = best_predictor.predict(test_features[:5])  # Only test 5 samples
    aligned_count = 0
    
    for pred in sample_predictions:
        class_range = best_predictor.class_score_ranges[pred.difficulty_class]
        if class_range[0] <= pred.difficulty_score <= class_range[1]:
            aligned_count += 1
    
    alignment_rate = aligned_count / len(sample_predictions)
    print(f"🎯 Prediction Alignment Rate: {alignment_rate:.1%}")
    
    # Save improved models
    print("\n💾 Saving improved models...")
    best_predictor.save_models()
    
    # Save improved feature extractor
    import joblib
    joblib.dump(feature_extractor, "models/improved_feature_extractor.joblib")
    
    # Save improved model metadata
    metadata = {
        "model_version": "improved_v1.0_fast",
        "training_samples": len(train_problems_split),
        "validation_samples": len(val_problems),
        "test_samples": len(test_problems_reduced),
        "features": train_features.shape[1],
        "best_algorithm": best_algorithm,
        "test_accuracy": test_class_metrics.accuracy,
        "test_r2": test_reg_metrics.r2_score,
        "test_mae": test_reg_metrics.mae,
        "test_rmse": test_reg_metrics.rmse,
        "accuracy_improvement": improvement,
        "alignment_rate": alignment_rate,
        "class_distribution": {
            class_name: test_class_reduced.count(class_name) 
            for class_name in ['Easy', 'Medium', 'Hard']
        },
        "training_time_minutes": (time.time() - start_time) / 60,
        "hyperparameter_tuning": False,  # Disabled for speed
        "feature_engineering": "enhanced",
        "prediction_alignment": True,
        "dataset_reduction": "30% train, 50% test",
        "fast_mode": True
    }
    
    with open("models/improved_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    training_time = (time.time() - start_time) / 60
    print(f"⏱️  Total training time: {training_time:.1f} minutes")
    print("✅ Improved models saved successfully!")
    
    # Success message
    if test_class_metrics.accuracy > 0.60:
        print("\n🎉 SUCCESS: Achieved >60% accuracy target!")
    elif test_class_metrics.accuracy > original_accuracy:
        print(f"\n📈 IMPROVEMENT: Accuracy increased by {improvement:.1f} percentage points!")
    else:
        print("\n⚠️  No significant improvement. Consider more data or different approaches.")
    
    print("\n📋 Next steps:")
    print("  1. Update Flask API to use improved models")
    print("  2. Test predictions with improved backend")
    print("  3. Compare results with original model")
    print("  4. Deploy improved model to production")


if __name__ == "__main__":
    main()