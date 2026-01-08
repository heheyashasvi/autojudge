"""
Improved feature extraction for AutoJudge ML system with enhanced accuracy.
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import logging
import nltk
from collections import Counter
import math

from .data_models import ProblemText, FeatureVector

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class ImprovedFeatureExtractor:
    """
    Enhanced feature extractor with better accuracy for programming problem difficulty prediction.
    """
    
    def __init__(self, max_features: int = 2000):
        """
        Initialize improved feature extractor.
        
        Args:
            max_features: Maximum number of TF-IDF features
        """
        self.max_features = max_features
        
        # Enhanced TF-IDF with better parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 3),  # Include trigrams for better context
            min_df=3,  # More restrictive minimum document frequency
            max_df=0.85,  # More restrictive maximum document frequency
            sublinear_tf=True,  # Use sublinear TF scaling
            norm='l2'  # L2 normalization
        )
        
        # Scaler for statistical features
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Enhanced programming concepts with difficulty weights
        self.algorithm_concepts = {
            # Easy concepts (weight 1)
            'array': 1, 'list': 1, 'string': 1, 'loop': 1, 'iteration': 1,
            'linear': 1, 'search': 1, 'sort': 1, 'basic': 1, 'simple': 1,
            
            # Medium concepts (weight 2)
            'hash': 2, 'map': 2, 'set': 2, 'stack': 2, 'queue': 2,
            'binary': 2, 'tree': 2, 'recursion': 2, 'divide': 2, 'conquer': 2,
            'greedy': 2, 'two': 2, 'pointer': 2, 'sliding': 2, 'window': 2,
            
            # Hard concepts (weight 3)
            'dynamic': 3, 'programming': 3, 'graph': 3, 'dfs': 3, 'bfs': 3,
            'dijkstra': 3, 'bellman': 3, 'ford': 3, 'floyd': 3, 'warshall': 3,
            'topological': 3, 'backtrack': 3, 'optimization': 3, 'minimum': 3,
            'spanning': 3, 'tree': 3, 'segment': 3, 'fenwick': 3, 'trie': 3,
            
            # Very hard concepts (weight 4)
            'suffix': 4, 'array': 4, 'kmp': 4, 'rabin': 4, 'karp': 4,
            'convex': 4, 'hull': 4, 'computational': 4, 'geometry': 4,
            'network': 4, 'flow': 4, 'bipartite': 4, 'matching': 4,
            'strongly': 4, 'connected': 4, 'components': 4
        }
        
        # Complexity indicators
        self.complexity_patterns = {
            r'O\(1\)': 1,           # Constant time
            r'O\(log\s*n\)': 2,     # Logarithmic
            r'O\(n\)': 2,           # Linear
            r'O\(n\s*log\s*n\)': 3, # N log N
            r'O\(n\^?2\)': 4,       # Quadratic
            r'O\(n\^?3\)': 5,       # Cubic
            r'O\(2\^?n\)': 6,       # Exponential
            r'O\(n!\)': 7,          # Factorial
        }
        
        # Mathematical symbols with weights
        self.math_symbols = {
            '+': 1, '-': 1, '*': 1, '/': 1, '=': 1,
            '<': 2, '>': 2, '≤': 2, '≥': 2, '!=': 2,
            '∑': 3, '∏': 3, '∫': 3, '√': 3, '^': 3,
            '²': 3, '³': 3, '∞': 4, 'π': 3, 'log': 3,
            'sin': 3, 'cos': 3, 'tan': 3, 'mod': 2
        }
        
        # Difficulty keywords
        self.difficulty_indicators = {
            'easy': -2, 'simple': -2, 'basic': -2, 'trivial': -3,
            'medium': 0, 'moderate': 0, 'intermediate': 0,
            'hard': 2, 'difficult': 2, 'complex': 2, 'challenging': 2,
            'advanced': 3, 'expert': 3, 'optimal': 2, 'efficient': 2,
            'minimize': 2, 'maximize': 2, 'optimization': 3
        }
    
    def extract_enhanced_statistical_features(self, problem: ProblemText) -> np.ndarray:
        """
        Extract enhanced statistical features with better difficulty discrimination.
        
        Args:
            problem: Problem text object
            
        Returns:
            Array of enhanced statistical features
        """
        combined_text = problem.get_combined_text().lower()
        title_text = (problem.title or '').lower()
        desc_text = (problem.description or '').lower()
        
        features = []
        
        # 1. Enhanced text length features
        features.append(len(combined_text))
        features.append(problem.get_word_count())
        features.append(len(title_text))
        features.append(len(desc_text))
        
        # 2. Text complexity metrics
        sentences = nltk.sent_tokenize(combined_text)
        features.append(len(sentences))  # Sentence count
        
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            features.append(avg_sentence_length)
        else:
            features.append(0)
        
        # 3. Vocabulary richness
        words = combined_text.split()
        unique_words = set(words)
        features.append(len(unique_words))  # Unique word count
        features.append(len(unique_words) / max(len(words), 1))  # Vocabulary richness ratio
        
        # 4. Weighted algorithm concept scoring
        algorithm_score = 0
        for concept, weight in self.algorithm_concepts.items():
            count = combined_text.count(concept)
            algorithm_score += count * weight
        features.append(algorithm_score)
        
        # 5. Complexity pattern detection
        complexity_score = 0
        for pattern, weight in self.complexity_patterns.items():
            matches = len(re.findall(pattern, combined_text, re.IGNORECASE))
            complexity_score += matches * weight
        features.append(complexity_score)
        
        # 6. Mathematical content analysis
        math_score = 0
        for symbol, weight in self.math_symbols.items():
            count = combined_text.count(symbol)
            math_score += count * weight
        features.append(math_score)
        
        # 7. Difficulty indicator scoring
        difficulty_score = 0
        for indicator, weight in self.difficulty_indicators.items():
            count = combined_text.count(indicator)
            difficulty_score += count * weight
        features.append(difficulty_score)
        
        # 8. Numeric analysis
        numbers = re.findall(r'\d+', combined_text)
        features.append(len(numbers))  # Number count
        
        if numbers:
            numeric_values = [int(n) for n in numbers if n.isdigit()]
            if numeric_values:
                features.append(max(numeric_values))  # Largest number
                features.append(np.mean(numeric_values))  # Average number
                features.append(len([n for n in numeric_values if n > 1000]))  # Large numbers
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0])
        
        # 9. Structural complexity
        features.append(combined_text.count('input'))
        features.append(combined_text.count('output'))
        features.append(combined_text.count('constraint'))
        features.append(combined_text.count('example'))
        features.append(combined_text.count('test'))
        features.append(combined_text.count('case'))
        
        # 10. Advanced patterns
        features.append(len(re.findall(r'\b\d+\s*≤\s*\w+\s*≤\s*\d+\b', combined_text)))  # Range constraints
        features.append(len(re.findall(r'\b\w+\[\w+\]', combined_text)))  # Array indexing
        features.append(len(re.findall(r'\b\w+\(\w*\)', combined_text)))  # Function calls
        
        # 11. Problem type indicators
        features.append(combined_text.count('shortest'))
        features.append(combined_text.count('minimum'))
        features.append(combined_text.count('maximum'))
        features.append(combined_text.count('optimal'))
        features.append(combined_text.count('path'))
        features.append(combined_text.count('distance'))
        
        # 12. Data structure mentions
        data_structures = ['array', 'list', 'tree', 'graph', 'stack', 'queue', 
                          'heap', 'hash', 'map', 'set', 'trie', 'segment']
        ds_count = sum(combined_text.count(ds) for ds in data_structures)
        features.append(ds_count)
        
        return np.array(features, dtype=float)
    
    def fit_transform(self, problems: List[ProblemText]) -> np.ndarray:
        """
        Fit the improved feature extractor and transform problems.
        
        Args:
            problems: List of problem text objects
            
        Returns:
            Enhanced feature matrix
        """
        # Extract combined text for TF-IDF
        combined_texts = [problem.get_combined_text() for problem in problems]
        
        # Fit and transform TF-IDF with enhanced parameters
        tfidf_features = self.tfidf_vectorizer.fit_transform(combined_texts).toarray()
        
        # Extract enhanced statistical features
        statistical_features = np.array([
            self.extract_enhanced_statistical_features(problem) 
            for problem in problems
        ])
        
        # Scale statistical features
        statistical_features_scaled = self.scaler.fit_transform(statistical_features)
        
        # Combine features
        combined_features = np.hstack([tfidf_features, statistical_features_scaled])
        
        self.is_fitted = True
        logger.info(f"Fitted improved feature extractor with {combined_features.shape[1]} features")
        logger.info(f"TF-IDF features: {tfidf_features.shape[1]}, Statistical features: {statistical_features.shape[1]}")
        
        return combined_features
    
    def transform(self, problems: List[ProblemText]) -> np.ndarray:
        """
        Transform problems using fitted improved extractor.
        
        Args:
            problems: List of problem text objects
            
        Returns:
            Enhanced feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        # Extract combined text for TF-IDF
        combined_texts = [problem.get_combined_text() for problem in problems]
        
        # Transform TF-IDF
        tfidf_features = self.tfidf_vectorizer.transform(combined_texts).toarray()
        
        # Extract enhanced statistical features
        statistical_features = np.array([
            self.extract_enhanced_statistical_features(problem) 
            for problem in problems
        ])
        
        # Scale statistical features
        statistical_features_scaled = self.scaler.transform(statistical_features)
        
        # Combine features
        combined_features = np.hstack([tfidf_features, statistical_features_scaled])
        
        return combined_features
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all enhanced features.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            return []
        
        # TF-IDF feature names
        tfidf_names = list(self.tfidf_vectorizer.get_feature_names_out())
        
        # Enhanced statistical feature names
        stat_names = [
            'total_chars', 'word_count', 'title_length', 'description_length',
            'sentence_count', 'avg_sentence_length', 'unique_words', 'vocab_richness',
            'algorithm_score', 'complexity_score', 'math_score', 'difficulty_score',
            'number_count', 'max_number', 'avg_number', 'large_numbers',
            'input_count', 'output_count', 'constraint_count', 'example_count',
            'test_count', 'case_count', 'range_constraints', 'array_indexing',
            'function_calls', 'shortest_count', 'minimum_count', 'maximum_count',
            'optimal_count', 'path_count', 'distance_count', 'data_structure_count'
        ]
        
        return tfidf_names + stat_names