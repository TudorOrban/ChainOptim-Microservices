param (
    [string]$buildCore = "false",
    [string]$buildNotifications = "false"
)

Clear-Host

# Navigate to the root directory where the core pom.xml and docker-compose.yml are located
Push-Location -Path ".."
Write-Host "Current directory: $(Get-Location)"

Write-Host "Stopping and removing current Docker containers..."
docker-compose -f docker-compose.yml down

# Core
if ($buildCore -eq "true") {
    Push-Location -Path "core"
    Write-Host "Current directory: $(Get-Location)"

    Write-Host "Building the Core project (skipping tests)..."
    mvn clean package -DskipTests

    Write-Host "Building Core Image..."
    docker build -t tudoraorban/chainoptim-core:latest .

    Write-Host "Upload Core Image to DockerHub..."
    docker push tudoraorban/chainoptim-core:latest

    Pop-Location
}

if ($buildNotifications -eq "true") {
    # Notifications
    Push-Location -Path "notifications"

    Write-Host "Current directory: $(Get-Location)"
    Write-Host "Building the Notifications project (skipping tests)..."
    mvn clean package -DskipTests

    Write-Host "Building Notifications Image..."
    docker build -t tudoraorban/chainoptim-notifications:latest .

    Write-Host "Upload Notifications Image to DockerHub..."
    docker push tudoraorban/chainoptim-notifications:latest

    Pop-Location
}

# Compose
Write-Host "Building and starting Docker containers..."
docker-compose up --build

