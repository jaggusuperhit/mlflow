import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
from sklearn.base import BaseEstimator

from mlProject.entity.config_entity import PredictionConfig
from mlProject import logger


class ModelPredictor:
    """
    Model prediction component for making predictions using a trained model.
    """
    
    def __init__(self, config: PredictionConfig):
        """
        Initialize the ModelPredictor component.
        
        Args:
            config: Configuration for model prediction
        """
        self.config = config
        self.model = self._load_model()
        self.feature_columns = config.feature_columns
        
    def _load_model(self) -> BaseEstimator:
        """
        Load the trained model.
        
        Returns:
            BaseEstimator: Loaded model
        """
        try:
            logger.info(f"Loading model from {self.config.model_path}")
            model = joblib.load(self.config.model_path)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _validate_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare input data for prediction.
        
        Args:
            data: Input data for prediction
            
        Returns:
            pd.DataFrame: Validated and prepared data
        """
        # Check if all required features are present
        missing_features = set(self.feature_columns) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing features in input data: {missing_features}")
        
        # Select only the required features in the correct order
        return data[self.feature_columns].copy()
    
    def predict(self, data: Union[pd.DataFrame, Dict[str, List[float]], List[List[float]]]) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            data: Input data for prediction. Can be a DataFrame, dictionary, or list of lists.
            
        Returns:
            np.ndarray: Predictions
        """
        try:
            # Convert input to DataFrame if needed
            if isinstance(data, dict):
                data = pd.DataFrame(data)
            elif isinstance(data, list):
                if not self.feature_columns:
                    raise ValueError("Feature columns not provided for list input")
                data = pd.DataFrame(data, columns=self.feature_columns)
            
            # Validate and prepare input data
            prepared_data = self._validate_input(data)
            
            # Make predictions
            logger.info(f"Making predictions with shape {prepared_data.shape}")
            predictions = self.model.predict(prepared_data)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise RuntimeError(f"Failed to make predictions: {e}")
    
    def predict_single(self, features: Dict[str, float]) -> float:
        """
        Make a prediction for a single sample.
        
        Args:
            features: Dictionary of feature name to value
            
        Returns:
            float: Prediction
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Make prediction
        predictions = self.predict(df)
        
        # Return the first prediction
        return float(predictions[0])
    
    def batch_predict(self, data_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Make predictions for a batch of data.
        
        Args:
            data_path: Path to the input data file (CSV)
            output_path: Path to save the predictions (CSV)
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        try:
            # Load data
            logger.info(f"Loading data from {data_path}")
            data = pd.read_csv(data_path)
            
            # Make predictions
            predictions = self.predict(data)
            
            # Add predictions to the data
            data['prediction'] = predictions
            
            # Save predictions if output path is provided
            if output_path:
                logger.info(f"Saving predictions to {output_path}")
                data.to_csv(output_path, index=False)
            
            return data
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise RuntimeError(f"Failed to perform batch prediction: {e}")
