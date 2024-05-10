param (
    [string]$rd = "false"
)

Push-Location -Path ".."
Write-Host "Current directory: $(Get-Location)"

Write-Host "Building the Demand project (skipping tests)..."
mvn clean package -DskipTests

Write-Host "Building Demand Image..."
docker build -t tudoraorban/chainoptim-demand:latest .

Write-Host "Upload Demand Image to DockerHub..."
docker push tudoraorban/chainoptim-demand:latest

if ($rd -eq "true") {
    kubectl rollout restart deployment chainoptim-demand
}

Pop-Location

