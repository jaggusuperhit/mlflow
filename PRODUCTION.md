# Production Deployment Guide

This guide provides instructions for setting up and running the ML pipeline in a production environment with MLflow tracking via DagHub.

## Prerequisites

- Python 3.8 or higher
- Virtual environment (venv, conda, etc.)
- DagHub account with access to the repository

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create and Activate a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n mlops_env python=3.8 -y
conda activate mlops_env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .  # Install the package in development mode
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory with the following content:

```
MLFLOW_TRACKING_URI=https://dagshub.com/jaggusuperhit/MLflow.mlflow
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password
```

Replace `your_username` and `your_password` with your DagHub credentials.

### 5. Test DagHub Connection

```bash
python test_dagshub_connection.py
```

This script will test the connection to DagHub and MLflow. If successful, you should see a message indicating that the test metric was logged.

## Running the Pipeline

### Run the Complete Pipeline

```bash
python run_pipeline.py
```

### Run Specific Stages

```bash
# Run only data ingestion (stage 1)
python run_pipeline.py --stages 1

# Run data ingestion and validation (stages 1 and 2)
python run_pipeline.py --stages 1 2

# Run model training and evaluation (stages 4 and 5)
python run_pipeline.py --stages 4 5
```

## Monitoring and Tracking

After running the pipeline, you can view the experiment results in DagHub:

1. Go to your DagHub repository: https://dagshub.com/jaggusuperhit/MLflow
2. Navigate to the "Experiments" tab
3. You should see your logged runs with metrics, parameters, and models

## Troubleshooting

### Authentication Issues

If you encounter authentication issues:

1. Verify your credentials in the `.env` file
2. Try logging in using the DagHub CLI:
   ```bash
   pip install dagshub
   dagshub login
   ```

### MLflow Connection Issues

If MLflow cannot connect to DagHub:

1. Check your internet connection
2. Verify the tracking URI is correct
3. Ensure your DagHub account has access to the repository

## Deployment Considerations

### Scheduled Runs

For scheduled runs, consider using tools like:
- Airflow
- GitHub Actions
- Cron jobs

### Containerization

For containerized deployment:
1. Use the provided Dockerfile
2. Build the image: `docker build -t mlops-project .`
3. Run with environment variables:
   ```bash
   docker run -e MLFLOW_TRACKING_URI=... -e MLFLOW_TRACKING_USERNAME=... -e MLFLOW_TRACKING_PASSWORD=... mlops-project
   ```

### Security Considerations

- Never commit the `.env` file to version control
- Consider using a secrets manager for production deployments
- Rotate your DagHub access tokens periodically
