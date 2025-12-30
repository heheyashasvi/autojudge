"""
Data models for the AutoJudge ML system.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class ProblemText:
    """
    Represents a programming problem with its textual components.
    """
    title: str
    description: str
    input_description: str
    output_description: str
    
    def get_combined_text(self) -> str:
        """
        Combine all text fields into a single string for processing.
        
        Returns:
            str: Combined text with all fields separated by spaces
        """
        # Handle None values gracefully
        title = self.title or ""
        description = self.description or ""
        input_desc = self.input_description or ""
        output_desc = self.output_description or ""
        
        # Combine with space separation
        combined = f"{title} {description} {input_desc} {output_desc}"
        
        # Clean up extra whitespace
        return " ".join(combined.split())
    
    def is_valid(self) -> bool:
        """
        Check if the problem text has sufficient content for analysis.
        
        Returns:
            bool: True if problem has meaningful content
        """
        combined = self.get_combined_text()
        return len(combined.strip()) > 0
    
    def get_word_count(self) -> int:
        """
        Get the total word count across all fields.
        
        Returns:
            int: Total number of words
        """
        combined = self.get_combined_text()
        return len(combined.split()) if combined.strip() else 0


@dataclass
class FeatureVector:
    """
    Represents extracted features from problem text.
    """
    tfidf_features: np.ndarray  # TF-IDF semantic features
    statistical_features: np.ndarray  # Length, keyword counts, etc.
    feature_names: List[str]  # Names of features for interpretability
    
    def to_array(self) -> np.ndarray:
        """
        Convert to a single numpy array for ML models.
        
        Returns:
            np.ndarray: Combined feature vector
        """
        return np.concatenate([self.tfidf_features, self.statistical_features])
    
    def get_feature_count(self) -> int:
        """
        Get the total number of features.
        
        Returns:
            int: Total feature count
        """
        return len(self.to_array())


@dataclass
class ClassificationMetrics:
    """
    Metrics for evaluating classification model performance.
    """
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    confusion_matrix: np.ndarray
    
    def get_macro_avg_f1(self) -> float:
        """
        Calculate macro-average F1 score.
        
        Returns:
            float: Macro-average F1 score
        """
        return np.mean(list(self.f1_score.values()))


@dataclass
class RegressionMetrics:
    """
    Metrics for evaluating regression model performance.
    """
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    r2_score: float  # R-squared
    
    def is_acceptable(self) -> bool:
        """
        Check if the regression metrics indicate acceptable performance.
        
        Returns:
            bool: True if metrics are within acceptable ranges
        """
        # Define acceptable thresholds
        return (self.mae < 2.0 and  # MAE less than 2 difficulty points
                self.rmse < 3.0 and  # RMSE less than 3 difficulty points
                self.r2_score > 0.5)  # R² greater than 0.5


@dataclass
class PredictionResult:
    """
    Result of difficulty prediction including both classification and regression.
    """
    difficulty_class: str  # Easy, Medium, Hard
    difficulty_score: float  # Numerical score
    confidence: Optional[float] = None  # Prediction confidence
    processing_time: Optional[float] = None  # Time taken for prediction
    
    def is_valid(self) -> bool:
        """
        Check if the prediction result is valid.
        
        Returns:
            bool: True if result is valid
        """
        valid_classes = {'Easy', 'Medium', 'Hard'}
        return (self.difficulty_class in valid_classes and
                0 <= self.difficulty_score <= 10 and
                np.isfinite(self.difficulty_score))


def create_problem_from_dict(data: Dict) -> ProblemText:
    """
    Create a ProblemText instance from a dictionary.
    
    Args:
        data: Dictionary with problem text fields
        
    Returns:
        ProblemText: Created problem instance
        
    Raises:
        KeyError: If required fields are missing
    """
    required_fields = ['title', 'description', 'input_description', 'output_description']
    
    # Check for required fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise KeyError(f"Missing required fields: {missing_fields}")
    
    return ProblemText(
        title=data['title'],
        description=data['description'],
        input_description=data['input_description'],
        output_description=data['output_description']
    )


def create_sample_problems() -> List[ProblemText]:
    """
    Create sample problems for testing and demonstration.
    
    Returns:
        List[ProblemText]: List of sample problems
    """
    samples = [
        ProblemText(
            title="Two Sum",
            description="Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
            input_description="An array of integers and a target integer",
            output_description="Array of two indices"
        ),
        ProblemText(
            title="Shortest Path in Weighted Graph",
            description="Find the shortest path from node A to all other nodes in a graph with non-negative weights using Dijkstra's algorithm.",
            input_description="V vertices, E edges with weights, source vertex",
            output_description="List of shortest distances from source to all vertices"
        ),
        ProblemText(
            title="Traveling Salesman Problem",
            description="Find the shortest possible route that visits every city exactly once and returns to the origin city. This is an NP-hard optimization problem.",
            input_description="Adjacency matrix of distances between N cities",
            output_description="Minimum weight of the optimal tour"
        )
    ]
    
    return samples