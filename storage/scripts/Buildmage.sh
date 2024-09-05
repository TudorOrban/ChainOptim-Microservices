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

# Build the Supply project (skipping tests)
echo "Building the Supply project (skipping tests)..."
mvn clean package -DskipTests

# Build Supply Docker image
echo "Building Supply Image..."
docker build -t tudoraorban/chainoptim-supply:latest .

# Upload Supply Docker image to DockerHub
echo "Upload Supply Image to DockerHub..."
docker push tudoraorban/chainoptim-supply:latest

# If rd is true, restart the deployment
if [ "$rd" = "true" ]; then
    echo "Restarting the Supply deployment..."
    kubectl rollout restart deployment chainoptim-supply
fi

# Return to the original directory
popd
