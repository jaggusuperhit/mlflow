# MLflow Integration with DagHub

This document provides instructions for using MLflow with DagHub for experiment tracking in this project.

## Setup

### 1. Create a DagHub Token

1. Log in to [DagHub](https://dagshub.com)
2. Go to Settings > Access Tokens
3. Create a new token with the following scopes:
   - `repo` (for repository access)
   - `mlflow` (for MLflow integration)
4. Copy the token for use in the next step

### 2. Configure Environment Variables

Create a `.env` file in the project root with the following content:

```
MLFLOW_TRACKING_URI=https://dagshub.com/jaggusuperhit/MLflow.mlflow
MLFLOW_TRACKING_USERNAME=jaggusuperhit
MLFLOW_TRACKING_PASSWORD=your_token_here
```

Replace `your_token_here` with the token you created in step 1.

### 3. Install Dependencies

Make sure you have the required dependencies installed:

```bash
pip install mlflow python-dotenv
```

## Usage

### Running the Pipeline

To run the entire pipeline including MLflow tracking:

```bash
python run_pipeline.py
```

To run only the model evaluation stage with MLflow tracking:

```bash
python run_pipeline.py --stages 5
```

### Viewing Experiments

To view your experiments:

1. Go to [https://dagshub.com/jaggusuperhit/MLflow](https://dagshub.com/jaggusuperhit/MLflow)
2. Navigate to the "Experiments" tab
3. You should see your experiments listed there

## Tracked Information

The following information is tracked in MLflow:

### Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R2 (R-squared score)

### Parameters
- alpha (ElasticNet regularization parameter)
- l1_ratio (ElasticNet mixing parameter)

### Tags
- model_type: "elasticnet"
- data_version: "1.0"

### Artifacts
- The trained model is saved as an MLflow artifact

## Troubleshooting

If you encounter issues with MLflow tracking:

1. Verify that your token is valid and has the correct permissions
2. Check that the environment variables are set correctly
3. Ensure you have network connectivity to DagHub
4. Look for error messages in the logs

## Advanced Usage

### Custom Experiments

To create a custom experiment, modify the `mlflow.set_experiment()` call in `src/mlProject/components/model_evaluation.py`.

### Additional Tracking

To track additional metrics, parameters, or artifacts, modify the `log_into_mlflow()` method in `src/mlProject/components/model_evaluation.py`.

### Programmatic Access

To access your MLflow experiments programmatically:

```python
import mlflow
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# List experiments
experiments = mlflow.search_experiments()
for exp in experiments:
    print(exp.name)

# Get runs from an experiment
runs = mlflow.search_runs(experiment_names=["wine-quality-prediction"])
print(runs)
```
