"""
Improved machine learning models for AutoJudge system with enhanced accuracy.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .data_models import ClassificationMetrics, RegressionMetrics, PredictionResult

logger = logging.getLogger(__name__)


class ImprovedDifficultyClassifier:
    """
    Enhanced Random Forest classifier with hyperparameter tuning for better accuracy.
    """
    
    def __init__(self, tune_hyperparameters: bool = True):
        """
        Initialize improved classifier.
        
        Args:
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        self.tune_hyperparameters = tune_hyperparameters
        self.model = None
        self.is_trained = False
        self.classes = ['Easy', 'Medium', 'Hard']
        self.label_encoder = LabelEncoder()
        
        # Best hyperparameters found through tuning (simplified for speed)
        self.best_params = {
            'n_estimators': 50,  # Reduced from 200
            'max_depth': 10,     # Reduced from 15
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'class_weight': 'balanced',
            'random_state': 42
        }
    
    def _tune_hyperparameters(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            features: Feature matrix
            labels: Encoded labels
            
        Returns:
            Best parameters
        """
        logger.info("Performing hyperparameter tuning for classifier...")
        
        # Parameter grid for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Base model
        base_model = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(features, labels)
        
        logger.info(f"Best accuracy: {grid_search.best_score_:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_params_
    
    def train(self, features: np.ndarray, labels: List[str]) -> None:
        """
        Train the improved classifier.
        
        Args:
            features: Feature matrix
            labels: Class labels
        """
        logger.info(f"Training improved classifier with {features.shape[0]} samples, {features.shape[1]} features")
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Hyperparameter tuning
        if self.tune_hyperparameters and features.shape[0] > 100:
            try:
                self.best_params.update(self._tune_hyperparameters(features, encoded_labels))
            except Exception as e:
                logger.warning(f"Hyperparameter tuning failed: {e}. Using default parameters.")
        
        # Train final model
        self.model = RandomForestClassifier(**self.best_params)
        self.model.fit(features, encoded_labels)
        self.is_trained = True
        
        # Log feature importance
        if hasattr(self.model, 'feature_importances_'):
            top_features = np.argsort(self.model.feature_importances_)[-10:]
            logger.info(f"Top 10 feature indices: {top_features}")
        
        logger.info("Improved classifier training completed")
    
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
        
        encoded_predictions = self.model.predict(features)
        predictions = self.label_encoder.inverse_transform(encoded_predictions)
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
        Evaluate improved classifier performance.
        
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


class ImprovedDifficultyRegressor:
    """
    Enhanced regressor with multiple algorithms and hyperparameter tuning.
    """
    
    def __init__(self, algorithm: str = 'gradient_boosting', tune_hyperparameters: bool = True):
        """
        Initialize improved regressor.
        
        Args:
            algorithm: Algorithm to use ('random_forest', 'gradient_boosting', 'svr')
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        self.algorithm = algorithm
        self.tune_hyperparameters = tune_hyperparameters
        self.model = None
        self.is_trained = False
        
        # Algorithm-specific best parameters (simplified for speed)
        self.best_params = {
            'random_forest': {
                'n_estimators': 50,  # Reduced from 200
                'max_depth': 10,     # Reduced from 15
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 50,  # Reduced from 150
                'learning_rate': 0.1,
                'max_depth': 4,      # Reduced from 6
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'subsample': 0.8,
                'random_state': 42
            },
            'svr': {
                'C': 10,
                'gamma': 'scale',
                'kernel': 'rbf',
                'epsilon': 0.1
            }
        }
    
    def _tune_hyperparameters(self, features: np.ndarray, scores: np.ndarray) -> Dict:
        """
        Perform hyperparameter tuning for the selected algorithm.
        
        Args:
            features: Feature matrix
            scores: Target scores
            
        Returns:
            Best parameters
        """
        logger.info(f"Performing hyperparameter tuning for {self.algorithm} regressor...")
        
        if self.algorithm == 'random_forest':
            param_grid = {
                'n_estimators': [100, 150, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            
        elif self.algorithm == 'gradient_boosting':
            param_grid = {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [4, 6, 8],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            }
            base_model = GradientBoostingRegressor(random_state=42)
            
        elif self.algorithm == 'svr':
            param_grid = {
                'C': [1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'poly'],
                'epsilon': [0.01, 0.1, 0.2]
            }
            base_model = SVR()
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(features, scores)
        
        logger.info(f"Best MAE: {-grid_search.best_score_:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_params_
    
    def train(self, features: np.ndarray, scores: List[float]) -> None:
        """
        Train the improved regressor.
        
        Args:
            features: Feature matrix
            scores: Difficulty scores
        """
        logger.info(f"Training improved {self.algorithm} regressor with {features.shape[0]} samples")
        
        scores_array = np.array(scores)
        
        # Hyperparameter tuning
        if self.tune_hyperparameters and features.shape[0] > 100:
            try:
                self.best_params[self.algorithm].update(
                    self._tune_hyperparameters(features, scores_array)
                )
            except Exception as e:
                logger.warning(f"Hyperparameter tuning failed: {e}. Using default parameters.")
        
        # Train final model
        if self.algorithm == 'random_forest':
            self.model = RandomForestRegressor(**self.best_params[self.algorithm])
        elif self.algorithm == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**self.best_params[self.algorithm])
        elif self.algorithm == 'svr':
            self.model = SVR(**self.best_params[self.algorithm])
        
        self.model.fit(features, scores_array)
        self.is_trained = True
        
        logger.info(f"Improved {self.algorithm} regressor training completed")
    
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
        Evaluate improved regressor performance.
        
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


class ImprovedAutoJudgePredictor:
    """
    Enhanced combined predictor with improved models and consistency alignment.
    """
    
    def __init__(self, model_dir: str = "models", regressor_algorithm: str = 'gradient_boosting'):
        """
        Initialize improved predictor.
        
        Args:
            model_dir: Directory to save/load models
            regressor_algorithm: Algorithm for regression ('random_forest', 'gradient_boosting', 'svr')
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.classifier = ImprovedDifficultyClassifier(tune_hyperparameters=True)
        self.regressor = ImprovedDifficultyRegressor(
            algorithm=regressor_algorithm, 
            tune_hyperparameters=True
        )
        self.feature_extractor = None
        
        # Consistency alignment parameters
        self.class_score_ranges = {
            'Easy': (0, 4),
            'Medium': (3, 7),
            'Hard': (6, 10)
        }
    
    def align_predictions(self, class_pred: str, score_pred: float) -> Tuple[str, float]:
        """
        Align classification and regression predictions for consistency.
        
        Args:
            class_pred: Predicted class
            score_pred: Predicted score
            
        Returns:
            Aligned (class, score) predictions
        """
        # Get expected score range for predicted class
        min_score, max_score = self.class_score_ranges[class_pred]
        
        # If score is outside expected range, adjust
        if score_pred < min_score or score_pred > max_score:
            # Find the class that best matches the score
            best_class = class_pred
            min_distance = float('inf')
            
            for cls, (cls_min, cls_max) in self.class_score_ranges.items():
                if cls_min <= score_pred <= cls_max:
                    best_class = cls
                    break
                else:
                    # Calculate distance to range
                    distance = min(abs(score_pred - cls_min), abs(score_pred - cls_max))
                    if distance < min_distance:
                        min_distance = distance
                        best_class = cls
            
            # If we changed the class, keep the original score
            # If we keep the class, adjust the score to be within range
            if best_class != class_pred:
                return best_class, score_pred
            else:
                adjusted_score = np.clip(score_pred, min_score, max_score)
                return class_pred, adjusted_score
        
        return class_pred, score_pred
    
    def train(self, features: np.ndarray, class_labels: List[str], scores: List[float]) -> Tuple[ClassificationMetrics, RegressionMetrics]:
        """
        Train both improved models.
        
        Args:
            features: Feature matrix
            class_labels: Difficulty class labels
            scores: Difficulty scores
            
        Returns:
            Tuple of (classification_metrics, regression_metrics)
        """
        logger.info("Training improved AutoJudge models...")
        
        # Train models
        self.classifier.train(features, class_labels)
        self.regressor.train(features, scores)
        
        # Evaluate on training data (for monitoring)
        class_metrics = self.classifier.evaluate(features, class_labels)
        reg_metrics = self.regressor.evaluate(features, scores)
        
        logger.info(f"Improved training completed - Classification accuracy: {class_metrics.accuracy:.3f}, Regression R²: {reg_metrics.r2_score:.3f}")
        
        return class_metrics, reg_metrics
    
    def predict(self, features: np.ndarray) -> List[PredictionResult]:
        """
        Make aligned predictions using both improved models.
        
        Args:
            features: Feature matrix
            
        Returns:
            List of prediction results with alignment
        """
        class_predictions = self.classifier.predict(features)
        score_predictions = self.regressor.predict(features)
        class_probabilities = self.classifier.predict_proba(features)
        
        results = []
        for i in range(len(class_predictions)):
            # Get confidence as max probability
            confidence = float(np.max(class_probabilities[i]))
            
            # Align predictions for consistency
            aligned_class, aligned_score = self.align_predictions(
                class_predictions[i], score_predictions[i]
            )
            
            result = PredictionResult(
                difficulty_class=aligned_class,
                difficulty_score=aligned_score,
                confidence=confidence
            )
            results.append(result)
        
        return results
    
    def save_models(self) -> None:
        """Save improved trained models to disk."""
        if not self.classifier.is_trained or not self.regressor.is_trained:
            raise ValueError("Models must be trained before saving")
        
        classifier_path = self.model_dir / "improved_classifier.joblib"
        regressor_path = self.model_dir / "improved_regressor.joblib"
        
        joblib.dump(self.classifier.model, classifier_path)
        joblib.dump(self.regressor.model, regressor_path)
        
        # Save label encoder
        encoder_path = self.model_dir / "label_encoder.joblib"
        joblib.dump(self.classifier.label_encoder, encoder_path)
        
        logger.info(f"Improved models saved to {self.model_dir}")
    
    def load_models(self) -> bool:
        """
        Load improved trained models from disk.
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        classifier_path = self.model_dir / "improved_classifier.joblib"
        regressor_path = self.model_dir / "improved_regressor.joblib"
        encoder_path = self.model_dir / "label_encoder.joblib"
        
        try:
            if classifier_path.exists() and regressor_path.exists() and encoder_path.exists():
                self.classifier.model = joblib.load(classifier_path)
                self.regressor.model = joblib.load(regressor_path)
                self.classifier.label_encoder = joblib.load(encoder_path)
                
                self.classifier.is_trained = True
                self.regressor.is_trained = True
                
                logger.info(f"Improved models loaded from {self.model_dir}")
                return True
            else:
                logger.warning(f"Improved model files not found in {self.model_dir}")
                return False
        except Exception as e:
            logger.error(f"Error loading improved models: {e}")
            return False