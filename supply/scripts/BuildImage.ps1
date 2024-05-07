Push-Location -Path ".."
Write-Host "Current directory: $(Get-Location)"

Write-Host "Building the Supply project (skipping tests)..."
mvn clean package -DskipTests

Write-Host "Building Supply Image..."
docker build -t tudoraorban/chainoptim-supply:latest .

Write-Host "Upload Notifications Image to DockerHub..."
docker push tudoraorban/chainoptim-supply:latest

Pop-Location

