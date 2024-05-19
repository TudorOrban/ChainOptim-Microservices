FROM openjdk:21-jdk-bullseye

WORKDIR /app

# Install mysql-client
RUN apt-get update && apt-get install -y default-mysql-client && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY target/notifications-1.0-SNAPSHOT.jar /app/ChainOptimizer.jar

ENTRYPOINT ["java", "-jar", "ChainOptimizer.jar"]

EXPOSE 8080
