# PowerShell script to test GitHub Actions workflow locally using act
# Requires act to be installed: https://github.com/nektos/act

# Check if act is installed
if (-not (Get-Command act -ErrorAction SilentlyContinue)) {
    Write-Error "act is not installed. Please install it from https://github.com/nektos/act"
    exit 1
}

# Create a .secrets file with your DockerHub credentials
if (-not (Test-Path .secrets)) {
    @"
DOCKERHUB_USERNAME=your_dockerhub_username
DOCKERHUB_TOKEN=your_dockerhub_token
"@ | Out-File -FilePath .secrets -Encoding utf8
    Write-Host "Created .secrets file. Please edit it with your actual DockerHub credentials."
    exit 0
}

# Run the test job
Write-Host "Running the test job..."
act -j test --secret-file .secrets

# If you want to run the build job (requires Docker)
# Write-Host "Running the build job..."
# act -j build --secret-file .secrets

# If you want to run the deploy job
# Write-Host "Running the deploy job..."
# act -j deploy --secret-file .secrets

Write-Host "Done!"
