param (
    [string]$rd = "false"
)

Push-Location -Path ".."
Write-Host "Current directory: $(Get-Location)"

Write-Host "Building the Notifications project (skipping tests)..."
mvn clean package -DskipTests

Write-Host "Building Notifications Image..."
docker build -t tudoraorban/chainoptim-notifications:latest .

Write-Host "Upload Notifications Image to DockerHub..."
docker push tudoraorban/chainoptim-notifications:latest

if ($rd -eq "true") {
    minikube rollout restart deployment chainoptim-notifications
}

Pop-Location

