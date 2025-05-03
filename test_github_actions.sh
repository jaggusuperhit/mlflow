#!/bin/bash
# Script to test GitHub Actions workflow locally using act
# Requires act to be installed: https://github.com/nektos/act

# Check if act is installed
if ! command -v act &> /dev/null; then
    echo "act is not installed. Please install it from https://github.com/nektos/act"
    exit 1
fi

# Create a .secrets file with your DockerHub credentials
if [ ! -f .secrets ]; then
    cat > .secrets << EOF
DOCKERHUB_USERNAME=your_dockerhub_username
DOCKERHUB_TOKEN=your_dockerhub_token
EOF
    echo "Created .secrets file. Please edit it with your actual DockerHub credentials."
    exit 0
fi

# Run the test job
echo "Running the test job..."
act -j test --secret-file .secrets

# If you want to run the build job (requires Docker)
# echo "Running the build job..."
# act -j build --secret-file .secrets

# If you want to run the deploy job
# echo "Running the deploy job..."
# act -j deploy --secret-file .secrets

echo "Done!"
