"""
Feature extraction for AutoJudge ML system.
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple
import logging

from .data_models import ProblemText, FeatureVector

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts features from programming problem text.
    """
    
    def __init__(self, max_features: int = 2000):
        """
        Initialize feature extractor.
        
        Args:
            max_features: Maximum number of TF-IDF features
        """
        self.max_features = max_features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        self.is_fitted = False
        
        # Programming keywords for counting
        self.programming_keywords = {
            'algorithm', 'sort', 'search', 'tree', 'graph', 'dynamic', 'programming',
            'recursion', 'iteration', 'array', 'list', 'stack', 'queue', 'hash',
            'binary', 'linear', 'logarithmic', 'polynomial', 'exponential',
            'greedy', 'divide', 'conquer', 'backtrack', 'optimization',
            'shortest', 'path', 'minimum', 'maximum', 'spanning', 'tree',
            'dijkstra', 'bellman', 'ford', 'floyd', 'warshall', 'kruskal',
            'prim', 'dfs', 'bfs', 'topological', 'sort'
        }
        
        # Mathematical symbols
        self.math_symbols = {
            '+', '-', '*', '/', '=', '<', '>', '≤', '≥', '∑', '∏', '∫',
            '√', '^', '²', '³', '∞', 'π', 'α', 'β', 'γ', 'δ', 'θ', 'λ'
        }
    
    def extract_statistical_features(self, problem: ProblemText) -> np.ndarray:
        """
        Extract statistical features from problem text.
        
        Args:
            problem: Problem text object
            
        Returns:
            Array of statistical features
        """
        combined_text = problem.get_combined_text().lower()
        
        features = []
        
        # Text length features
        features.append(len(combined_text))  # Total character count
        features.append(problem.get_word_count())  # Word count
        features.append(len(problem.title) if problem.title else 0)  # Title length
        features.append(len(problem.description) if problem.description else 0)  # Description length
        
        # Complexity indicators
        features.append(combined_text.count('algorithm'))
        features.append(combined_text.count('complexity'))
        features.append(combined_text.count('time'))
        features.append(combined_text.count('space'))
        
        # Programming keyword count
        keyword_count = sum(1 for keyword in self.programming_keywords 
                          if keyword in combined_text)
        features.append(keyword_count)
        
        # Mathematical symbol count
        math_count = sum(combined_text.count(symbol) for symbol in self.math_symbols)
        features.append(math_count)
        
        # Structural features
        features.append(combined_text.count('input'))
        features.append(combined_text.count('output'))
        features.append(combined_text.count('example'))
        features.append(combined_text.count('constraint'))
        
        # Difficulty indicators
        features.append(combined_text.count('optimal'))
        features.append(combined_text.count('efficient'))
        features.append(combined_text.count('minimum'))
        features.append(combined_text.count('maximum'))
        
        # Numeric patterns
        number_pattern = r'\d+'
        numbers = re.findall(number_pattern, combined_text)
        features.append(len(numbers))  # Count of numbers
        
        # Large number indicators (often indicate complexity)
        large_numbers = [int(n) for n in numbers if n.isdigit() and int(n) > 1000]
        features.append(len(large_numbers))
        
        return np.array(features, dtype=float)
    
    def fit_transform(self, problems: List[ProblemText]) -> np.ndarray:
        """
        Fit the feature extractor and transform problems.
        
        Args:
            problems: List of problem text objects
            
        Returns:
            Feature matrix
        """
        # Extract combined text for TF-IDF
        combined_texts = [problem.get_combined_text() for problem in problems]
        
        # Fit and transform TF-IDF
        tfidf_features = self.tfidf_vectorizer.fit_transform(combined_texts).toarray()
        
        # Extract statistical features
        statistical_features = np.array([
            self.extract_statistical_features(problem) 
            for problem in problems
        ])
        
        # Combine features
        combined_features = np.hstack([tfidf_features, statistical_features])
        
        self.is_fitted = True
        logger.info(f"Fitted feature extractor with {combined_features.shape[1]} features")
        
        return combined_features
    
    def transform(self, problems: List[ProblemText]) -> np.ndarray:
        """
        Transform problems using fitted extractor.
        
        Args:
            problems: List of problem text objects
            
        Returns:
            Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        # Extract combined text for TF-IDF
        combined_texts = [problem.get_combined_text() for problem in problems]
        
        # Transform TF-IDF
        tfidf_features = self.tfidf_vectorizer.transform(combined_texts).toarray()
        
        # Extract statistical features
        statistical_features = np.array([
            self.extract_statistical_features(problem) 
            for problem in problems
        ])
        
        # Combine features
        combined_features = np.hstack([tfidf_features, statistical_features])
        
        return combined_features
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all features.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            return []
        
        # TF-IDF feature names
        tfidf_names = list(self.tfidf_vectorizer.get_feature_names_out())
        
        # Statistical feature names
        stat_names = [
            'total_chars', 'word_count', 'title_length', 'description_length',
            'algorithm_count', 'complexity_count', 'time_count', 'space_count',
            'programming_keywords', 'math_symbols', 'input_count', 'output_count',
            'example_count', 'constraint_count', 'optimal_count', 'efficient_count',
            'minimum_count', 'maximum_count', 'number_count', 'large_number_count'
        ]
        
        return tfidf_names + stat_names