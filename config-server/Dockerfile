FROM openjdk:21-jdk-bullseye

WORKDIR /app

COPY target/chainoptim-config-server-0.0.1-SNAPSHOT.jar /app/ChainOptimizer.jar

ENTRYPOINT ["java", "-jar", "ChainOptimizer.jar"]

EXPOSE 8888
