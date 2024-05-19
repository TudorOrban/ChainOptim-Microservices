#!/bin/bash

# Default value for rd
rd="false"

# Parsing command line arguments for 'rd'
while getopts ":r:" opt; do
  case $opt in
    r) rd="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# Navigate to parent directory
pushd ..

# Display the current directory
echo "Current directory: $(pwd)"

# Build the API gateway project (skipping tests)
echo "Building the API gateway project (skipping tests)..."
mvn clean package -DskipTests

# Build API gateway Docker image
echo "Building API gateway Image..."
docker build -t tudoraorban/chainoptim-apigateway:latest .

# Upload API gateway Docker image to DockerHub
echo "Upload API gateway Image to DockerHub..."
docker push tudoraorban/chainoptim-apigateway:latest

# If rd is true, restart the deployment
if [ "$rd" = "true" ]; then
    echo "Restarting the apigateway deployment..."
    kubectl rollout restart deployment chainoptim-apigateway
fi

# Return to the original directory
popd
