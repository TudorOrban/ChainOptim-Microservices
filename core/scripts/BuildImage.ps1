param (
    [string]$rd = "false"
)

Push-Location -Path ".."
Write-Host "Current directory: $(Get-Location)"

Write-Host "Building the Core project (skipping tests)..."
mvn clean package -DskipTests

Write-Host "Building Core Image..."
docker build -t tudoraorban/chainoptim-core:latest .

Write-Host "Upload Core Image to DockerHub..."
docker push tudoraorban/chainoptim-core:latest

if ($rd -eq "true") {
    kubectl rollout restart deployment chainoptim-core
}

Pop-Location

