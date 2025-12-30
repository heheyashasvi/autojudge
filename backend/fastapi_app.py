"""
FastAPI server for AutoJudge ML model endpoints.
High-performance alternative to Flask with automatic API documentation.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import joblib
import numpy as np
from pathlib import Path
import time
from typing import Optional
import json

from ml.data_models import ProblemText, create_problem_from_dict
from ml.models import AutoJudgePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="AutoJudge ML API",
    description="Machine Learning API for programming problem difficulty prediction",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc"  # ReDoc at /redoc
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class ProblemRequest(BaseModel):
    """Request model for problem difficulty prediction."""
    title: str = Field(..., description="Problem title", example="Two Sum")
    description: str = Field(..., description="Problem description", 
                           example="Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.")
    input_description: str = Field(..., description="Input format description",
                                 example="An array of integers and a target integer")
    output_description: str = Field(..., description="Output format description",
                                  example="Array of two indices")

class PredictionResponse(BaseModel):
    """Response model for difficulty prediction."""
    difficulty_class: str = Field(..., description="Predicted difficulty class", example="Easy")
    difficulty_score: float = Field(..., description="Predicted difficulty score (0-10)", example=3.2)
    confidence: Optional[float] = Field(None, description="Prediction confidence (0-1)", example=0.85)
    processing_time: float = Field(..., description="Processing time in seconds", example=0.123)
    success: bool = Field(..., description="Whether prediction was successful", example=True)

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., example="healthy")
    service: str = Field(..., example="autojudge-ml-backend")
    models_loaded: bool = Field(..., example=True)

class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_info: dict = Field(..., description="Model metadata and performance metrics")
    success: bool = Field(..., example=True)

class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error message")
    success: bool = Field(..., example=False)

# Global variables for models
predictor = None
feature_extractor = None
model_loaded = False

def load_models():
    """Load trained models on startup."""
    global predictor, feature_extractor, model_loaded
    
    try:
        models_dir = Path("models")
        if not models_dir.exists():
            logger.error("Models directory not found. Please train models first.")
            return False
        
        # Load predictor
        predictor = AutoJudgePredictor()
        if not predictor.load_models():
            logger.error("Failed to load ML models")
            return False
        
        # Load feature extractor
        feature_extractor_path = models_dir / "feature_extractor.joblib"
        if feature_extractor_path.exists():
            feature_extractor = joblib.load(feature_extractor_path)
            logger.info("Feature extractor loaded successfully")
        else:
            logger.error("Feature extractor not found")
            return False
        
        model_loaded = True
        logger.info("✅ All models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load models when the application starts."""
    logger.info("🚀 Starting AutoJudge FastAPI Backend...")
    logger.info("📦 Loading trained models...")
    
    if load_models():
        logger.info("✅ Models loaded successfully!")
    else:
        logger.error("❌ Failed to load models. Please train models first: python train_models.py")

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status of the API and whether models are loaded.
    """
    return HealthResponse(
        status="healthy",
        service="autojudge-ml-backend",
        models_loaded=model_loaded
    )

@app.post("/predict", response_model=PredictionResponse, responses={
    400: {"model": ErrorResponse, "description": "Bad Request"},
    503: {"model": ErrorResponse, "description": "Service Unavailable"},
    500: {"model": ErrorResponse, "description": "Internal Server Error"}
}, tags=["Prediction"])
async def predict_difficulty(request: ProblemRequest):
    """
    Predict difficulty for a programming problem using trained ML models.
    
    This endpoint processes the problem text through feature extraction and 
    returns both classification (Easy/Medium/Hard) and regression (0-10 score) predictions.
    
    - **title**: The problem title
    - **description**: Detailed problem description
    - **input_description**: Description of input format and constraints
    - **output_description**: Description of expected output format
    
    Returns difficulty class, numerical score, confidence, and processing time.
    """
    start_time = time.time()
    
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="ML models not loaded. Please check server logs."
        )
    
    try:
        # Create problem text object
        problem_data = request.dict()
        problem = create_problem_from_dict(problem_data)
        
        # Validate problem has content
        if not problem.is_valid():
            raise HTTPException(
                status_code=400,
                detail="Problem text is empty or invalid"
            )
        
        # Extract features
        try:
            features = feature_extractor.transform([problem])
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Feature extraction failed"
            )
        
        # Make prediction
        try:
            predictions = predictor.predict(features)
            result = predictions[0]  # Get first (and only) prediction
            
            processing_time = time.time() - start_time
            
            return PredictionResponse(
                difficulty_class=result.difficulty_class,
                difficulty_score=round(result.difficulty_score, 2),
                confidence=round(result.confidence, 3) if result.confidence else None,
                processing_time=round(processing_time, 3),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Prediction failed"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@app.get("/model-info", response_model=ModelInfoResponse, responses={
    503: {"model": ErrorResponse, "description": "Service Unavailable"}
}, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded ML models.
    
    Returns model metadata including training statistics, performance metrics,
    and dataset information.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded"
        )
    
    try:
        # Load metadata
        metadata_path = Path("models/metadata.json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {"message": "No metadata available"}
        
        return ModelInfoResponse(
            model_info=metadata,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get model info"
        )

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "AutoJudge ML API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "predict": "/predict",
        "model_info": "/model-info"
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting AutoJudge FastAPI Backend...")
    print("📖 API Documentation available at:")
    print("   - Swagger UI: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print("🌐 Starting server on http://localhost:8000")
    
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )