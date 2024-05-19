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

# Build the Config Server project (skipping tests)
echo "Building the Config Server project (skipping tests)..."
mvn clean package -DskipTests

# Build Config Server Docker image
echo "Building Config Server Image..."
docker build -t tudoraorban/chainoptim-config-server:latest .

# Upload Config Server Docker image to DockerHub
echo "Upload Config Server Image to DockerHub..."
docker push tudoraorban/chainoptim-config-server:latest

# If rd is true, restart the deployment
if [ "$rd" = "true" ]; then
    echo "Restarting the Notifications deployment..."
    kubectl rollout restart deployment chainoptim-config-server
fi

# Return to the original directory
popd
