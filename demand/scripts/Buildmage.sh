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

# Build the Demand project (skipping tests)
echo "Building the Demand project (skipping tests)..."
mvn clean package -DskipTests

# Build Demand Docker image
echo "Building Demand Image..."
docker build -t tudoraorban/chainoptim-demand:latest .

# Upload Demand Docker image to DockerHub
echo "Upload Demand Image to DockerHub..."
docker push tudoraorban/chainoptim-demand:latest

# If rd is true, restart the deployment
if [ "$rd" = "true" ]; then
    echo "Restarting the Demand deployment..."
    kubectl rollout restart deployment chainoptim-demand
fi

# Return to the original directory
popd
