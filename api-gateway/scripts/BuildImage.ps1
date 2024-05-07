Push-Location -Path ".."
Write-Host "Current directory: $(Get-Location)"

Write-Host "Building the API Gateway project (skipping tests)..."
mvn clean package -DskipTests

Write-Host "Building Gateway Image..."
docker build -t tudoraorban/chainoptim-apigateway:latest .

Write-Host "Upload Gateway Image to DockerHub..."
# docker push tudoraorban/chainoptim-apigateway:latest

Pop-Location

