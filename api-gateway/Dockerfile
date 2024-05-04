# Use a base image with OpenJDK installed
FROM openjdk:21-jdk-bullseye

# Set the working directory inside the container
WORKDIR /app

# Update packages and install any necessary packages
# Assuming no additional packages are needed beyond the base image
RUN apt-get update && apt-get install -y curl && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the built jar file from the build context into the container
COPY target/api-gateway-1.0-SNAPSHOT.jar /app/api-gateway.jar

# Specify the command to run the application
ENTRYPOINT ["java", "-jar", "api-gateway.jar"]

# Expose the port on which the gateway will run
EXPOSE 8082
