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

# Build the Core project (skipping tests)
echo "Building the Core project (skipping tests)..."
mvn clean package -DskipTests

# Build Core Docker image
echo "Building Core Image..."
docker build -t tudoraorban/chainoptim-core:latest .

# Upload Core Docker image to DockerHub
echo "Upload Core Image to DockerHub..."
docker push tudoraorban/chainoptim-core:latest

# If rd is true, restart the deployment
if [ "$rd" = "true" ]; then
    echo "Restarting the Core deployment..."
    kubectl rollout restart deployment chainoptim-core
fi

# Return to the original directory
popd
