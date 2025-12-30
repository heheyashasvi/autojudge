"""
Dataset loader for AutoJudge ML system.
Supports loading programming problem datasets in JSONL format.
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import asdict

from .data_models import ProblemText, create_problem_from_dict

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads and processes programming problem datasets.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize dataset loader.
        
        Args:
            dataset_path: Path to the JSONL dataset file
        """
        self.dataset_path = Path(dataset_path)
        self.problems: List[ProblemText] = []
        self.labels: Dict[str, List] = {}  # Store both class and score labels
        self.raw_data: List[Dict] = []
        
    def load_jsonl(self) -> List[Dict]:
        """
        Load data from JSONL file.
        
        Returns:
            List[Dict]: List of problem dictionaries
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            json.JSONDecodeError: If JSONL format is invalid
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        data = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(data)} items from {self.dataset_path}")
        self.raw_data = data
        return data
    
    def detect_schema(self, sample_size: int = 10) -> Dict[str, str]:
        """
        Detect the schema of the dataset by examining sample records.
        
        Args:
            sample_size: Number of records to examine
            
        Returns:
            Dict mapping expected fields to detected fields
        """
        if not self.raw_data:
            self.load_jsonl()
        
        if not self.raw_data:
            return {}
        
        # Examine first few records
        sample = self.raw_data[:min(sample_size, len(self.raw_data))]
        
        # Common field name variations
        field_mappings = {
            'title': ['title', 'name', 'problem_name', 'problem_title'],
            'description': ['description', 'problem_description', 'statement', 'problem_statement', 'text'],
            'input_description': ['input_description', 'input_format', 'input', 'input_spec'],
            'output_description': ['output_description', 'output_format', 'output', 'output_spec'],
            'problem_class': ['problem_class', 'difficulty', 'difficulty_class', 'level', 'class'],
            'problem_score': ['problem_score', 'difficulty_score', 'score', 'rating', 'difficulty_rating']
        }
        
        detected_schema = {}
        all_keys = set()
        
        # Collect all keys from sample
        for item in sample:
            all_keys.update(item.keys())
        
        # Find best matches
        for expected_field, possible_names in field_mappings.items():
            for possible_name in possible_names:
                if possible_name in all_keys:
                    detected_schema[expected_field] = possible_name
                    break
        
        logger.info(f"Detected schema: {detected_schema}")
        logger.info(f"Available fields in dataset: {sorted(all_keys)}")
        
        return detected_schema
    
    def map_difficulty_classes(self, raw_class) -> str:
        """
        Map various difficulty representations to standard classes.
        
        Args:
            raw_class: Raw difficulty value from dataset
            
        Returns:
            str: Standardized difficulty class (Easy/Medium/Hard)
        """
        if raw_class is None:
            return "Medium"  # Default
        
        raw_str = str(raw_class).lower().strip()
        
        # Direct mappings
        if raw_str in ['easy', 'e', '1', 'beginner', 'simple']:
            return "Easy"
        elif raw_str in ['medium', 'm', '2', 'intermediate', 'moderate']:
            return "Medium"
        elif raw_str in ['hard', 'h', '3', 'difficult', 'advanced', 'expert']:
            return "Hard"
        
        # Numeric mappings (common in competitive programming)
        try:
            numeric_val = float(raw_class)
            if numeric_val <= 1200:
                return "Easy"
            elif numeric_val <= 1800:
                return "Medium"
            else:
                return "Hard"
        except (ValueError, TypeError):
            pass
        
        # Default to Medium if can't determine
        logger.warning(f"Could not map difficulty class: {raw_class}, defaulting to Medium")
        return "Medium"
    
    def normalize_score(self, raw_score, min_score: float = 0, max_score: float = 10) -> float:
        """
        Normalize difficulty scores to 0-10 range.
        
        Args:
            raw_score: Raw score from dataset
            min_score: Minimum value for output range
            max_score: Maximum value for output range
            
        Returns:
            float: Normalized score between min_score and max_score
        """
        if raw_score is None:
            return 5.0  # Default middle value
        
        try:
            score = float(raw_score)
            
            # If already in 0-10 range, return as-is
            if 0 <= score <= 10:
                return score
            
            # Common competitive programming ranges
            if 800 <= score <= 3500:  # Codeforces-style rating
                # Map 800-3500 to 0-10
                normalized = (score - 800) / (3500 - 800) * (max_score - min_score) + min_score
                return max(min_score, min(max_score, normalized))
            
            # If score is 1-5 range, map to 0-10
            if 1 <= score <= 5:
                return (score - 1) / 4 * (max_score - min_score) + min_score
            
            # Default normalization: assume it's already roughly in correct range
            return max(min_score, min(max_score, score))
            
        except (ValueError, TypeError):
            logger.warning(f"Could not normalize score: {raw_score}, defaulting to 5.0")
            return 5.0
    
    def process_dataset(self, schema_mapping: Optional[Dict[str, str]] = None) -> Tuple[List[ProblemText], Dict[str, List]]:
        """
        Process the loaded dataset into ProblemText objects and labels.
        
        Args:
            schema_mapping: Optional manual schema mapping
            
        Returns:
            Tuple of (problems, labels) where labels contains 'class' and 'score' lists
        """
        if not self.raw_data:
            self.load_jsonl()
        
        if schema_mapping is None:
            schema_mapping = self.detect_schema()
        
        problems = []
        class_labels = []
        score_labels = []
        
        for item in self.raw_data:
            try:
                # Extract text fields
                problem_data = {
                    'title': item.get(schema_mapping.get('title', 'title'), ''),
                    'description': item.get(schema_mapping.get('description', 'description'), ''),
                    'input_description': item.get(schema_mapping.get('input_description', 'input_description'), ''),
                    'output_description': item.get(schema_mapping.get('output_description', 'output_description'), '')
                }
                
                # Create problem text object
                problem = create_problem_from_dict(problem_data)
                
                # Skip if problem has no meaningful content
                if not problem.is_valid():
                    continue
                
                problems.append(problem)
                
                # Extract labels
                raw_class = item.get(schema_mapping.get('problem_class', 'problem_class'))
                raw_score = item.get(schema_mapping.get('problem_score', 'problem_score'))
                
                difficulty_class = self.map_difficulty_classes(raw_class)
                difficulty_score = self.normalize_score(raw_score)
                
                class_labels.append(difficulty_class)
                score_labels.append(difficulty_score)
                
            except Exception as e:
                logger.warning(f"Skipping problematic record: {e}")
                continue
        
        self.problems = problems
        self.labels = {
            'class': class_labels,
            'score': score_labels
        }
        
        logger.info(f"Processed {len(problems)} valid problems")
        logger.info(f"Class distribution: {pd.Series(class_labels).value_counts().to_dict()}")
        logger.info(f"Score range: {min(score_labels):.2f} - {max(score_labels):.2f}")
        
        return problems, self.labels
    
    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[List[ProblemText], List[ProblemText], Dict[str, List], Dict[str, List]]:
        """
        Split dataset into training and testing sets.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_problems, test_problems, train_labels, test_labels)
        """
        if not self.problems:
            self.process_dataset()
        
        from sklearn.model_selection import train_test_split
        
        # Split the data
        train_problems, test_problems, train_class, test_class = train_test_split(
            self.problems, self.labels['class'], 
            test_size=test_size, random_state=random_state, stratify=self.labels['class']
        )
        
        train_scores, test_scores = train_test_split(
            self.labels['score'], test_size=test_size, random_state=random_state
        )
        
        train_labels = {'class': train_class, 'score': train_scores}
        test_labels = {'class': test_class, 'score': test_scores}
        
        return train_problems, test_problems, train_labels, test_labels
    
    def save_processed_data(self, output_path: str):
        """
        Save processed data to a new JSONL file.
        
        Args:
            output_path: Path to save processed data
        """
        if not self.problems:
            self.process_dataset()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, problem in enumerate(self.problems):
                record = {
                    **asdict(problem),
                    'problem_class': self.labels['class'][i],
                    'problem_score': self.labels['score'][i]
                }
                f.write(json.dumps(record) + '\n')
        
        logger.info(f"Saved {len(self.problems)} processed records to {output_path}")


def create_sample_dataset(output_path: str, num_samples: int = 100):
    """
    Create a sample dataset for testing purposes.
    
    Args:
        output_path: Path to save sample dataset
        num_samples: Number of sample problems to generate
    """
    import random
    
    # Sample problem templates
    templates = [
        {
            "title": "Two Sum",
            "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
            "input_description": "An array of integers and a target integer",
            "output_description": "Array of two indices",
            "problem_class": "Easy",
            "problem_score": 2.5
        },
        {
            "title": "Longest Palindromic Substring",
            "description": "Given a string s, return the longest palindromic substring in s.",
            "input_description": "A string s of length n",
            "output_description": "The longest palindromic substring",
            "problem_class": "Medium",
            "problem_score": 5.5
        },
        {
            "title": "N-Queens",
            "description": "The n-queens puzzle is the problem of placing n queens on an n×n chessboard such that no two queens attack each other.",
            "input_description": "An integer n",
            "output_description": "All distinct solutions to the n-queens puzzle",
            "problem_class": "Hard",
            "problem_score": 8.5
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            template = random.choice(templates)
            # Add some variation
            record = template.copy()
            record['title'] = f"{record['title']} Variant {i+1}"
            record['problem_score'] += random.uniform(-1, 1)
            record['problem_score'] = max(0, min(10, record['problem_score']))
            
            f.write(json.dumps(record) + '\n')
    
    print(f"Created sample dataset with {num_samples} problems at {output_path}")


if __name__ == "__main__":
    # Example usage
    create_sample_dataset("sample_dataset.jsonl", 50)
    
    loader = DatasetLoader("sample_dataset.jsonl")
    problems, labels = loader.process_dataset()
    
    print(f"Loaded {len(problems)} problems")
    print(f"Class distribution: {pd.Series(labels['class']).value_counts()}")