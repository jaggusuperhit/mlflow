# CI/CD Pipeline Setup Guide

This guide explains how to set up the GitHub Actions CI/CD pipeline for this project.

## Required GitHub Secrets

The CI/CD pipeline requires the following secrets to be set up in your GitHub repository:

1. **DOCKERHUB_USERNAME**: Your DockerHub username
2. **DOCKERHUB_TOKEN**: Your DockerHub access token (not your password)

### How to Set Up GitHub Secrets

1. Go to your GitHub repository
2. Click on "Settings" tab
3. In the left sidebar, click on "Secrets and variables" > "Actions"
4. Click on "New repository secret"
5. Add each of the required secrets:

#### DockerHub Credentials

1. Create a DockerHub access token:
   - Log in to [DockerHub](https://hub.docker.com/)
   - Go to Account Settings > Security
   - Click "New Access Token"
   - Give it a name (e.g., "GitHub Actions")
   - Copy the token value

2. Add the secrets to GitHub:
   - Add `DOCKERHUB_USERNAME` with your DockerHub username
   - Add `DOCKERHUB_TOKEN` with the access token you created

## CI/CD Pipeline Workflow

The CI/CD pipeline consists of three main jobs:

1. **Test**: Runs unit tests and generates coverage reports
   - Triggered on all pushes to main and all pull requests
   - Uses pytest to run tests and generate coverage reports
   - Uploads coverage reports to Codecov

2. **Build**: Builds and pushes the Docker image
   - Only runs after tests pass
   - Only triggered on pushes to main branch
   - Builds the Docker image and pushes it to DockerHub
   - Tags the image with both `latest` and the commit SHA

3. **Deploy**: Deploys the application (currently a placeholder)
   - Only runs after build succeeds
   - Only triggered on pushes to main branch
   - Currently just prints a message
   - Contains commented code for Kubernetes deployment

## Kubernetes Deployment

The deployment job is currently set up as a placeholder. When you're ready to deploy to Kubernetes:

1. Uncomment the kubectl commands in the deployment job
2. Set up any additional secrets needed for Kubernetes authentication
3. Configure the Kubernetes context and namespace as needed

## Troubleshooting

If the CI/CD pipeline fails, check the following:

1. **Test failures**: Look at the test logs to see which tests failed
2. **DockerHub authentication**: Verify your DockerHub credentials are correct
3. **Docker build issues**: Check if the Dockerfile is valid and all dependencies are available
4. **Deployment issues**: Ensure your Kubernetes configuration is correct

## Local Testing

You can test the CI/CD pipeline locally before pushing to GitHub:

1. Run tests: `pytest tests/ --cov=src/mlProject`
2. Build Docker image: `docker build -t jaggusuperhit/wine-quality-api:local .`
3. Run the Docker image: `docker run -p 8000:8000 jaggusuperhit/wine-quality-api:local`

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [DockerHub Documentation](https://docs.docker.com/docker-hub/)
- [Kubernetes Documentation](https://kubernetes.io/docs/home/)
