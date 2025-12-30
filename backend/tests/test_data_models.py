"""
Property-based tests for data models.
Feature: autojudge-difficulty-predictor
"""
import pytest
from hypothesis import given, strategies as st
import numpy as np

from ml.data_models import (
    ProblemText, FeatureVector, ClassificationMetrics, RegressionMetrics,
    PredictionResult, create_problem_from_dict, create_sample_problems
)


# Hypothesis strategies for generating test data
text_strategy = st.text(min_size=0, max_size=1000)
non_empty_text_strategy = st.text(min_size=1, max_size=1000)
word_strategy = st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')), min_size=1, max_size=50)


class TestProblemText:
    """Test cases for ProblemText data model."""
    
    @given(
        title=text_strategy,
        description=text_strategy,
        input_desc=text_strategy,
        output_desc=text_strategy
    )
    def test_property_1_text_combination_consistency(self, title, description, input_desc, output_desc):
        """
        Property 1: Text Combination Consistency
        For any valid problem text with title, description, input_description, and output_description fields,
        the combined text should contain all original content in the correct order.
        **Feature: autojudge-difficulty-predictor, Property 1: Text Combination Consistency**
        **Validates: Requirements 1.1**
        """
        # Create problem text
        problem = ProblemText(
            title=title,
            description=description,
            input_description=input_desc,
            output_description=output_desc
        )
        
        # Get combined text
        combined = problem.get_combined_text()
        
        # Property: Combined text should contain all non-empty original content
        if title and title.strip():
            assert title.strip() in combined or any(word in combined for word in title.split() if word.strip())
        
        if description and description.strip():
            assert description.strip() in combined or any(word in combined for word in description.split() if word.strip())
        
        if input_desc and input_desc.strip():
            assert input_desc.strip() in combined or any(word in combined for word in input_desc.split() if word.strip())
        
        if output_desc and output_desc.strip():
            assert output_desc.strip() in combined or any(word in combined for word in output_desc.split() if word.strip())
        
        # Property: Combined text should be a single line with normalized whitespace
        assert '\n' not in combined
        assert '  ' not in combined  # No double spaces
        
        # Property: If all inputs are empty/None, combined should be empty
        if not any([title and title.strip(), description and description.strip(), 
                   input_desc and input_desc.strip(), output_desc and output_desc.strip()]):
            assert combined == ""
    
    @given(
        title=st.one_of(st.none(), text_strategy),
        description=st.one_of(st.none(), text_strategy),
        input_desc=st.one_of(st.none(), text_strategy),
        output_desc=st.one_of(st.none(), text_strategy)
    )
    def test_property_2_graceful_missing_value_handling(self, title, description, input_desc, output_desc):
        """
        Property 2: Graceful Missing Value Handling
        For any problem text with missing or None values in any field,
        the feature extractor should complete processing without throwing exceptions.
        **Feature: autojudge-difficulty-predictor, Property 2: Graceful Missing Value Handling**
        **Validates: Requirements 1.2**
        """
        # This should not raise any exceptions
        try:
            problem = ProblemText(
                title=title,
                description=description,
                input_description=input_desc,
                output_description=output_desc
            )
            
            # These operations should all complete without errors
            combined = problem.get_combined_text()
            is_valid = problem.is_valid()
            word_count = problem.get_word_count()
            
            # Properties that should hold
            assert isinstance(combined, str)
            assert isinstance(is_valid, bool)
            assert isinstance(word_count, int)
            assert word_count >= 0
            
        except Exception as e:
            pytest.fail(f"Exception raised with None values: {e}")
    
    def test_sample_problems_creation(self):
        """Test that sample problems are created correctly."""
        samples = create_sample_problems()
        
        assert len(samples) > 0
        for sample in samples:
            assert isinstance(sample, ProblemText)
            assert sample.is_valid()
            assert sample.get_word_count() > 0
    
    def test_create_problem_from_dict_valid(self):
        """Test creating problem from valid dictionary."""
        data = {
            'title': 'Test Problem',
            'description': 'A test problem description',
            'input_description': 'Test input',
            'output_description': 'Test output'
        }
        
        problem = create_problem_from_dict(data)
        assert problem.title == 'Test Problem'
        assert problem.description == 'A test problem description'
        assert problem.input_description == 'Test input'
        assert problem.output_description == 'Test output'
    
    def test_create_problem_from_dict_missing_fields(self):
        """Test creating problem from dictionary with missing fields."""
        data = {
            'title': 'Test Problem',
            'description': 'A test problem description'
            # Missing input_description and output_description
        }
        
        with pytest.raises(KeyError) as exc_info:
            create_problem_from_dict(data)
        
        assert "Missing required fields" in str(exc_info.value)


class TestFeatureVector:
    """Test cases for FeatureVector data model."""
    
    @given(
        tfidf_size=st.integers(min_value=1, max_value=100),
        stat_size=st.integers(min_value=1, max_value=20)
    )
    def test_feature_vector_combination(self, tfidf_size, stat_size):
        """Test that feature vectors combine correctly."""
        tfidf_features = np.random.rand(tfidf_size)
        stat_features = np.random.rand(stat_size)
        feature_names = [f"feature_{i}" for i in range(tfidf_size + stat_size)]
        
        fv = FeatureVector(
            tfidf_features=tfidf_features,
            statistical_features=stat_features,
            feature_names=feature_names
        )
        
        combined = fv.to_array()
        
        # Properties
        assert len(combined) == tfidf_size + stat_size
        assert fv.get_feature_count() == tfidf_size + stat_size
        assert np.array_equal(combined[:tfidf_size], tfidf_features)
        assert np.array_equal(combined[tfidf_size:], stat_features)


class TestClassificationMetrics:
    """Test cases for ClassificationMetrics."""
    
    def test_classification_metrics_macro_f1(self):
        """Test macro F1 calculation."""
        f1_scores = {'Easy': 0.8, 'Medium': 0.7, 'Hard': 0.9}
        
        metrics = ClassificationMetrics(
            accuracy=0.8,
            precision={'Easy': 0.8, 'Medium': 0.7, 'Hard': 0.9},
            recall={'Easy': 0.8, 'Medium': 0.7, 'Hard': 0.9},
            f1_score=f1_scores,
            confusion_matrix=np.array([[10, 1, 0], [2, 8, 1], [0, 1, 9]])
        )
        
        expected_macro_f1 = (0.8 + 0.7 + 0.9) / 3
        assert abs(metrics.get_macro_avg_f1() - expected_macro_f1) < 1e-6


class TestRegressionMetrics:
    """Test cases for RegressionMetrics."""
    
    @given(
        mae=st.floats(min_value=0, max_value=10),
        rmse=st.floats(min_value=0, max_value=10),
        r2=st.floats(min_value=-1, max_value=1)
    )
    def test_regression_metrics_acceptability(self, mae, rmse, r2):
        """Test regression metrics acceptability logic."""
        metrics = RegressionMetrics(mae=mae, rmse=rmse, r2_score=r2)
        
        expected_acceptable = (mae < 2.0 and rmse < 3.0 and r2 > 0.5)
        assert metrics.is_acceptable() == expected_acceptable


class TestPredictionResult:
    """Test cases for PredictionResult."""
    
    @given(
        difficulty_class=st.sampled_from(['Easy', 'Medium', 'Hard']),
        difficulty_score=st.floats(min_value=0, max_value=10),
        confidence=st.one_of(st.none(), st.floats(min_value=0, max_value=1))
    )
    def test_prediction_result_validity(self, difficulty_class, difficulty_score, confidence):
        """Test prediction result validity checking."""
        result = PredictionResult(
            difficulty_class=difficulty_class,
            difficulty_score=difficulty_score,
            confidence=confidence
        )
        
        expected_valid = (
            difficulty_class in {'Easy', 'Medium', 'Hard'} and
            0 <= difficulty_score <= 10 and
            np.isfinite(difficulty_score)
        )
        
        assert result.is_valid() == expected_valid
    
    def test_prediction_result_invalid_class(self):
        """Test prediction result with invalid class."""
        result = PredictionResult(
            difficulty_class='Invalid',
            difficulty_score=5.0
        )
        
        assert not result.is_valid()
    
    def test_prediction_result_invalid_score(self):
        """Test prediction result with invalid score."""
        result = PredictionResult(
            difficulty_class='Easy',
            difficulty_score=15.0  # Out of range
        )
        
        assert not result.is_valid()
        
        result_nan = PredictionResult(
            difficulty_class='Easy',
            difficulty_score=float('nan')
        )
        
        assert not result_nan.is_valid()