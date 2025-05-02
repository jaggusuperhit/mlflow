import os
import sys
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import joblib
import mlflow
from pathlib import Path
import time
import logging

from src.mlProject.components.model_prediction import ModelPredictor
from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.utils.common import read_yaml
from src.mlProject import logger

# Initialize FastAPI app
app = FastAPI(
    title="Wine Quality Prediction API",
    description="API for predicting wine quality using ElasticNet model",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class PredictionRequest(BaseModel):
    fixed_acidity: float = Field(..., description="Fixed acidity")
    volatile_acidity: float = Field(..., description="Volatile acidity")
    citric_acid: float = Field(..., description="Citric acid")
    residual_sugar: float = Field(..., description="Residual sugar")
    chlorides: float = Field(..., description="Chlorides")
    free_sulfur_dioxide: float = Field(..., description="Free sulfur dioxide")
    total_sulfur_dioxide: float = Field(..., description="Total sulfur dioxide")
    density: float = Field(..., description="Density")
    pH: float = Field(..., description="pH")
    sulphates: float = Field(..., description="Sulphates")
    alcohol: float = Field(..., description="Alcohol")

    class Config:
        schema_extra = {
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4
            }
        }

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    prediction_time: float

class BatchPredictionRequest(BaseModel):
    data: List[Dict[str, float]]

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
    prediction_time: float

class HealthResponse(BaseModel):
    status: str
    model_version: str
    api_version: str

# Global variables
MODEL_PATH = "artifacts/model_trainer/model.joblib"
MODEL_VERSION = "1.0.0"
predictor = None

# Dependency to get predictor
def get_predictor():
    global predictor
    if predictor is None:
        try:
            # Initialize configuration manager
            config_manager = ConfigurationManager()
            prediction_config = config_manager.get_prediction_config(MODEL_PATH)

            # Initialize predictor
            predictor = ModelPredictor(prediction_config)
            logger.info("Model predictor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing predictor: {e}")
            raise HTTPException(status_code=500, detail=f"Error initializing model: {str(e)}")
    return predictor

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request {request.method} {request.url.path} processed in {process_time:.4f} seconds")
    return response

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# API endpoints
@app.get("/", response_model=HealthResponse)
def read_root():
    """Root endpoint to check if the API is running."""
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "api_version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "api_version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, predictor: ModelPredictor = Depends(get_predictor)):
    """
    Make a prediction for a single wine sample.
    """
    try:
        start_time = time.time()

        # Convert request to dictionary
        features = request.dict()

        # Make prediction
        prediction = predictor.predict_single(features)

        prediction_time = time.time() - start_time

        return {
            "prediction": prediction,
            "model_version": MODEL_VERSION,
            "prediction_time": prediction_time
        }
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest, predictor: ModelPredictor = Depends(get_predictor)):
    """
    Make predictions for multiple wine samples.
    """
    try:
        start_time = time.time()

        # Convert request to DataFrame
        df = pd.DataFrame(request.data)

        # Make predictions
        predictions = predictor.predict(df).tolist()

        prediction_time = time.time() - start_time

        return {
            "predictions": predictions,
            "model_version": MODEL_VERSION,
            "prediction_time": prediction_time
        }
    except Exception as e:
        logger.error(f"Error making batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Run the application
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run the FastAPI app
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)