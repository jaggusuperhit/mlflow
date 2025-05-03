# Wine Quality Prediction - MLOps Project

## Overview

This project demonstrates MLOps best practices for a wine quality prediction model. It includes:

- Modular, production-grade code structure
- Data ingestion, validation, and transformation pipeline
- Model training and evaluation
- MLflow integration for experiment tracking
- RESTful API using FastAPI
- Docker containerization
- Unit testing

## Project Structure

```
├── .dockerignore              # Files to exclude from Docker build
├── .env                       # Environment variables (not in git)
├── .env.example               # Example environment variables
├── .gitignore                 # Files to exclude from git
├── Dockerfile                 # Docker configuration
├── README.md                  # Project documentation
├── app.py                     # FastAPI application
├── config/                    # Configuration files
│   ├── config.yaml            # Main configuration
│   └── schema.yaml            # Data schema definition
├── main.py                    # Main training pipeline
├── params.yaml                # Model hyperparameters
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Project dependencies
├── research/                  # Jupyter notebooks for research
├── setup.py                   # Package setup
├── src/                       # Source code
│   └── mlProject/             # Main package
│       ├── __init__.py        # Package initialization
│       ├── components/        # Pipeline components
│       ├── config/            # Configuration management
│       ├── constants/         # Constants and paths
│       ├── entity/            # Data classes
│       ├── pipeline/          # Pipeline orchestration
│       └── utils/             # Utility functions
└── tests/                     # Unit and integration tests
    └── test_model_prediction.py  # Tests for prediction component
```

## Installation

### Option 1: Local Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/jaggusuperhit/MLflow.git
   cd MLflow
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Copy example env file and modify with your credentials
   cp .env.example .env
   # Edit .env with your preferred editor
   ```

### Option 2: Using Docker

1. Clone the repository:

   ```bash
   git clone https://github.com/jaggusuperhit/MLflow.git
   cd MLflow
   ```

2. Build the Docker image:

   ```bash
   docker build -t wine-quality-prediction .
   ```

3. Run the Docker container:
   ```bash
   docker run -p 8000:8000 --env-file .env wine-quality-prediction
   ```

## Usage

### Running the Training Pipeline

To train the model and track with MLflow:

```bash
python main.py
```

This will:

1. Ingest the wine quality dataset
2. Validate the data schema
3. Perform data transformations
4. Train an ElasticNet model
5. Evaluate the model
6. Log metrics and the model to MLflow

### Using the API

Start the API server:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000 with the following endpoints:

- `GET /`: Health check
- `GET /docs`: Interactive API documentation (Swagger UI)
- `POST /predict`: Make a prediction for a single wine sample
- `POST /batch-predict`: Make predictions for multiple wine samples

### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
         }'
```

## MLflow Integration

This project uses MLflow for experiment tracking and model registry. The model training pipeline automatically logs parameters, metrics, and the model to MLflow.

### DagHub Integration

The project is configured to use DagHub as the MLflow tracking server. To set up DagHub:

1. Create an account on [DagHub](https://dagshub.com/)
2. Create a new repository
3. Get your authentication token
4. Set environment variables:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/jaggusuperhit/MLflow.mlflow
export MLFLOW_TRACKING_USERNAME=jaggusuperhit
export MLFLOW_TRACKING_PASSWORD=your_token
```

Or use the Python API:

```python
import dagshub
dagshub.init(repo_owner='jaggusuperhit', repo_name='MLflow', mlflow=True)

import mlflow
with mlflow.start_run():
    mlflow.log_param('parameter_name', value)
    mlflow.log_metric('metric_name', value)
    mlflow.sklearn.log_model(model, "model")
```

## CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment. The pipeline:

1. Runs tests and generates coverage reports
2. Builds and pushes a Docker image to DockerHub
3. Deploys the application (currently a placeholder)

For more details on setting up the CI/CD pipeline, see [CICD_SETUP.md](CICD_SETUP.md).

### Testing the CI/CD Pipeline Locally

You can test the GitHub Actions workflow locally using [act](https://github.com/nektos/act):

```bash
# On Linux/Mac
./test_github_actions.sh

# On Windows
./test_github_actions.ps1
```

## Testing

Run tests with pytest:

```bash
pytest
```

For test coverage:

```bash
pytest --cov=src --cov-report=html
```

## Project Workflow

1. Data Ingestion: Download and extract the wine quality dataset
2. Data Validation: Validate the schema of the dataset
3. Data Transformation: Preprocess the data for model training
4. Model Training: Train an ElasticNet regression model
5. Model Evaluation: Evaluate the model on test data and log metrics to MLflow
6. Model Serving: Serve the model via a RESTful API

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or feedback, please contact:

- Email: jagtapsuraj636@gmail.com
- GitHub: [jaggusuperhit](https://github.com/jaggusuperhit)
