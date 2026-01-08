"""
Machine learning models for AutoJudge system.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from .data_models import ClassificationMetrics, RegressionMetrics, PredictionResult

logger = logging.getLogger(__name__)


class DifficultyClassifier:
    """
    Random Forest classifier for difficulty prediction.
    """
    
    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        """
        Initialize classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',  # Handle class imbalance
            oob_score=True  # Enable OOB for stacking
        )
        self.is_trained = False
        self.classes = ['Easy', 'Medium', 'Hard']
    
    def train(self, features: np.ndarray, labels: List[str]) -> None:
        """
        Train the classifier.
        
        Args:
            features: Feature matrix
            labels: Class labels
        """
        logger.info(f"Training classifier with {features.shape[0]} samples, {features.shape[1]} features")
        
        self.model.fit(features, labels)
        self.is_trained = True
        
        logger.info("Classifier training completed")
    
    def predict(self, features: np.ndarray) -> List[str]:
        """
        Predict difficulty classes.
        
        Args:
            features: Feature matrix
            
        Returns:
            List of predicted classes
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.model.predict(features)
        return predictions.tolist()
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            features: Feature matrix
            
        Returns:
            Probability matrix
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(features)
    
    def evaluate(self, test_features: np.ndarray, test_labels: List[str]) -> ClassificationMetrics:
        """
        Evaluate classifier performance.
        
        Args:
            test_features: Test feature matrix
            test_labels: True test labels
            
        Returns:
            Classification metrics
        """
        predictions = self.predict(test_features)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(test_labels, predictions, output_dict=True)
        cm = confusion_matrix(test_labels, predictions, labels=self.classes)
        
        # Extract per-class metrics
        precision = {cls: report[cls]['precision'] for cls in self.classes if cls in report}
        recall = {cls: report[cls]['recall'] for cls in self.classes if cls in report}
        f1_score = {cls: report[cls]['f1-score'] for cls in self.classes if cls in report}
        
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            confusion_matrix=cm
        )
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns:
            Feature importance array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        return self.model.feature_importances_


class DifficultyRegressor:
    """
    Gradient Boosting regressor for difficulty score prediction.
    """
    
    def __init__(self, n_estimators: int = 500, random_state: int = 42):
        """
        Initialize regressor.
        
        Args:
            n_estimators: Number of boosting stages to perform
            random_state: Random seed for reproducibility
        """
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            loss='squared_error'
        )
        self.is_trained = False
    
    def train(self, features: np.ndarray, scores: List[float]) -> None:
        """
        Train the regressor.
        
        Args:
            features: Feature matrix
            scores: Difficulty scores
        """
        logger.info(f"Training regressor with {features.shape[0]} samples, {features.shape[1]} features")
        
        self.model.fit(features, scores)
        self.is_trained = True
        
        logger.info("Regressor training completed")
    
    def predict(self, features: np.ndarray) -> List[float]:
        """
        Predict difficulty scores.
        
        Args:
            features: Feature matrix
            
        Returns:
            List of predicted scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.model.predict(features)
        # Ensure predictions are in valid range [0, 10]
        predictions = np.clip(predictions, 0, 10)
        return predictions.tolist()
    
    def evaluate(self, test_features: np.ndarray, test_scores: List[float]) -> RegressionMetrics:
        """
        Evaluate regressor performance.
        
        Args:
            test_features: Test feature matrix
            test_scores: True test scores
            
        Returns:
            Regression metrics
        """
        predictions = self.predict(test_features)
        
        # Calculate metrics
        mae = mean_absolute_error(test_scores, predictions)
        rmse = np.sqrt(mean_squared_error(test_scores, predictions))
        r2 = r2_score(test_scores, predictions)
        
        return RegressionMetrics(
            mae=mae,
            rmse=rmse,
            r2_score=r2
        )
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns:
            Feature importance array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        return self.model.feature_importances_


class AutoJudgePredictor:
    """
    Combined predictor that uses both classification and regression models.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize predictor.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.classifier = DifficultyClassifier()
        self.regressor = DifficultyRegressor()
        self.feature_extractor = None
        
    def train(self, features: np.ndarray, class_labels: List[str], scores: List[float]) -> Tuple[ClassificationMetrics, RegressionMetrics]:
        """
        Train both models with stacking.
        
        Args:
            features: Feature matrix
            class_labels: Difficulty class labels
            scores: Difficulty scores
            
        Returns:
            Tuple of (classification_metrics, regression_metrics)
        """
        # Train classifier first
        self.classifier.train(features, class_labels)
        
        # STACKING: Use classifier outputs as features for regressor
        # During training, we use Out-of-Bag (OOB) estimates to prevent data leakage/overfitting
        # If OOB is not available (e.g. not enough trees), we fallback to predict_proba (less ideal but functional)
        try:
            if hasattr(self.classifier.model, "oob_decision_function_"):
                logger.info("Using OOB estimates for regressor stacking")
                class_probs = self.classifier.model.oob_decision_function_
            else:
                logger.warning("OOB not available, using standard prediction for stacking (risk of overfitting)")
                class_probs = self.classifier.predict_proba(features)
        except Exception as e:
            logger.warning(f"Stacking error: {e}, using standard prediction")
            class_probs = self.classifier.predict_proba(features)
            
        # Augment features: [TF-IDF Features] + [Prob_Easy, Prob_Medium, Prob_Hard]
        features_aug = np.hstack((features, class_probs))
        
        # Train regressor on augmented features
        logger.info(f"Training regressor on stacked features (Shape: {features_aug.shape})")
        self.regressor.train(features_aug, scores)
        
        # Evaluate 
        # Note: For evaluation, we use predict_proba because that's what we'll use in production
        test_probs = self.classifier.predict_proba(features)
        features_eval = np.hstack((features, test_probs))
        
        class_metrics = self.classifier.evaluate(features, class_labels)
        reg_metrics = self.regressor.evaluate(features_eval, scores)
        
        logger.info(f"Training completed - Classification accuracy: {class_metrics.accuracy:.3f}, Regression R²: {reg_metrics.r2_score:.3f}")
        
        return class_metrics, reg_metrics
    
    def predict(self, features: np.ndarray) -> List[PredictionResult]:
        """
        Make predictions using both models with stacking.
        
        Args:
            features: Feature matrix
            
        Returns:
            List of prediction results
        """
        # 1. Get Classifier predictions
        class_predictions = self.classifier.predict(features)
        class_probabilities = self.classifier.predict_proba(features)
        
        # 2. Augment features with class probabilities
        features_aug = np.hstack((features, class_probabilities))
        
        # 3. Get Regressor predictions using augmented features
        score_predictions = self.regressor.predict(features_aug)
        
        results = []
        for i in range(len(class_predictions)):
            # Get confidence as max probability
            confidence = float(np.max(class_probabilities[i]))
            
            result = PredictionResult(
                difficulty_class=class_predictions[i],
                difficulty_score=score_predictions[i],
                confidence=confidence
            )
            results.append(result)
        
        return results
    
    def save_models(self) -> None:
        """Save trained models to disk."""
        if not self.classifier.is_trained or not self.regressor.is_trained:
            raise ValueError("Models must be trained before saving")
        
        classifier_path = self.model_dir / "classifier.joblib"
        regressor_path = self.model_dir / "regressor.joblib"
        
        joblib.dump(self.classifier.model, classifier_path)
        joblib.dump(self.regressor.model, regressor_path)
        
        logger.info(f"Models saved to {self.model_dir}")
    
    def load_models(self) -> bool:
        """
        Load trained models from disk.
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        classifier_path = self.model_dir / "classifier.joblib"
        regressor_path = self.model_dir / "regressor.joblib"
        
        try:
            if classifier_path.exists() and regressor_path.exists():
                self.classifier.model = joblib.load(classifier_path)
                self.regressor.model = joblib.load(regressor_path)
                
                self.classifier.is_trained = True
                self.regressor.is_trained = True
                
                logger.info(f"Models loaded from {self.model_dir}")
                return True
            else:
                logger.warning(f"Model files not found in {self.model_dir}")
                return False
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False