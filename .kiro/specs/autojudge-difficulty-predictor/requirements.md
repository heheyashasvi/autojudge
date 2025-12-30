# Requirements Document

## Introduction

AutoJudge is an intelligent system that automatically predicts the difficulty of programming problems based solely on their textual descriptions. The system performs both classification (Easy/Medium/Hard) and regression (numerical difficulty score) tasks, providing predictions through a simple web interface. This eliminates the need for human judgment and user feedback in the difficulty assessment process.

## Glossary

- **Problem_Description**: The main textual explanation of a programming problem
- **Input_Description**: Specification of the input format and constraints
- **Output_Description**: Specification of the expected output format
- **Problem_Class**: Categorical difficulty level (Easy, Medium, Hard)
- **Problem_Score**: Numerical difficulty rating
- **Feature_Extractor**: Component that converts text into numerical features
- **Classification_Model**: Machine learning model that predicts problem class
- **Regression_Model**: Machine learning model that predicts problem score
- **Web_Interface**: User-facing component for inputting problems and displaying predictions
- **Predictor_System**: The complete system that processes input and returns predictions

## Requirements

### Requirement 1: Data Processing and Feature Extraction

**User Story:** As a data scientist, I want to process raw problem text into numerical features, so that machine learning models can analyze the content effectively.

#### Acceptance Criteria

1. WHEN problem text fields are provided, THE Feature_Extractor SHALL combine title, description, input_description, and output_description into a single text input
2. WHEN text contains missing values, THE Feature_Extractor SHALL handle them gracefully without causing system failure
3. THE Feature_Extractor SHALL extract text length as a numerical feature
4. THE Feature_Extractor SHALL count mathematical symbols and programming keywords as features
5. THE Feature_Extractor SHALL generate TF-IDF vectors from the combined text
6. WHEN feature extraction is complete, THE Feature_Extractor SHALL output a consistent numerical feature vector

### Requirement 2: Classification Model Implementation

**User Story:** As a system user, I want to predict whether a problem is Easy, Medium, or Hard, so that I can understand the categorical difficulty level.

#### Acceptance Criteria

1. THE Classification_Model SHALL predict problem_class as one of three categories: Easy, Medium, or Hard
2. WHEN a feature vector is provided, THE Classification_Model SHALL return a single class prediction
3. THE Classification_Model SHALL achieve reasonable accuracy on the validation dataset
4. WHEN invalid input is provided, THE Classification_Model SHALL handle errors gracefully
5. THE Classification_Model SHALL use only textual features for prediction

### Requirement 3: Regression Model Implementation

**User Story:** As a system user, I want to predict a numerical difficulty score, so that I can get a precise difficulty assessment beyond categorical classification.

#### Acceptance Criteria

1. THE Regression_Model SHALL predict a numerical problem_score
2. WHEN a feature vector is provided, THE Regression_Model SHALL return a single numerical score
3. THE Regression_Model SHALL achieve acceptable error metrics (MAE/RMSE) on validation data
4. WHEN invalid input is provided, THE Regression_Model SHALL handle errors gracefully
5. THE Regression_Model SHALL use only textual features for prediction

### Requirement 4: Model Evaluation and Validation

**User Story:** As a developer, I want to evaluate model performance, so that I can ensure the system provides reliable predictions.

#### Acceptance Criteria

1. WHEN evaluating the classification model, THE Predictor_System SHALL calculate accuracy metrics
2. WHEN evaluating the classification model, THE Predictor_System SHALL generate a confusion matrix
3. WHEN evaluating the regression model, THE Predictor_System SHALL calculate Mean Absolute Error (MAE)
4. WHEN evaluating the regression model, THE Predictor_System SHALL calculate Root Mean Square Error (RMSE)
5. THE Predictor_System SHALL validate models on held-out test data

### Requirement 5: Web Interface for Problem Input

**User Story:** As a user, I want to input problem descriptions through a web interface, so that I can easily get difficulty predictions for new problems.

#### Acceptance Criteria

1. THE Web_Interface SHALL provide text input fields for problem description, input description, and output description
2. WHEN a user enters problem text, THE Web_Interface SHALL accept multi-line text input
3. THE Web_Interface SHALL provide a "Predict" button to trigger the prediction process
4. WHEN input fields are empty, THE Web_Interface SHALL handle the case gracefully
5. THE Web_Interface SHALL be accessible through a standard web browser

### Requirement 6: Prediction Results Display

**User Story:** As a user, I want to see predicted difficulty class and score, so that I can understand the system's assessment of the problem difficulty.

#### Acceptance Criteria

1. WHEN prediction is complete, THE Web_Interface SHALL display the predicted difficulty class (Easy/Medium/Hard)
2. WHEN prediction is complete, THE Web_Interface SHALL display the predicted numerical difficulty score
3. THE Web_Interface SHALL format results in a clear, readable manner
4. WHEN prediction fails, THE Web_Interface SHALL display an appropriate error message
5. THE Web_Interface SHALL update results immediately after prediction completion

### Requirement 7: System Integration and Processing Pipeline

**User Story:** As a system architect, I want all components to work together seamlessly, so that the end-to-end prediction process functions reliably.

#### Acceptance Criteria

1. WHEN a user submits problem text, THE Predictor_System SHALL process it through the complete pipeline from input to prediction
2. THE Predictor_System SHALL coordinate between feature extraction, classification, and regression components
3. WHEN processing is complete, THE Predictor_System SHALL return both class and score predictions
4. THE Predictor_System SHALL handle errors at any stage without crashing
5. THE Predictor_System SHALL process predictions within a reasonable time frame

### Requirement 8: Model Persistence and Loading

**User Story:** As a developer, I want trained models to be saved and loaded efficiently, so that the system can make predictions without retraining.

#### Acceptance Criteria

1. WHEN models are trained, THE Predictor_System SHALL save them to persistent storage
2. WHEN the system starts, THE Predictor_System SHALL load pre-trained models automatically
3. THE Predictor_System SHALL handle cases where model files are missing or corrupted
4. WHEN loading models, THE Predictor_System SHALL verify model compatibility with current feature extraction
5. THE Predictor_System SHALL support model versioning for future updates