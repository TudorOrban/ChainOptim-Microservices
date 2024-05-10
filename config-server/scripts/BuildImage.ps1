param (
    [string]$rd = "false"
)

Push-Location -Path ".."
Write-Host "Current directory: $(Get-Location)"

Write-Host "Building the Config Server project (skipping tests)..."
mvn clean package -DskipTests

Write-Host "Building Config Server Image..."
docker build -t tudoraorban/chainoptim-config-server:latest .

Write-Host "Upload Config Server Image to DockerHub..."
docker push tudoraorban/chainoptim-config-server:latest

if ($rd -eq "true") {
    kubectl rollout restart deployment chainoptim-config-server
}

Pop-Location

