FROM openjdk:21-jdk-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y curl && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY target/api-gateway-1.0-SNAPSHOT.jar /app/api-gateway.jar

ENTRYPOINT ["java", "-jar", "api-gateway.jar"]

EXPOSE 8082
