### ChainOptim

ChainOptim is a supply chain manager and optimizer. This repository aims to migrate the ChainOptim backend that will be used in production to a microservices architecture. For the monolithic version and a breakdown of the features, please refer to [this repo](https://github.com/TudorOrban/ChainOptim-backend).

Below is a diagram of the architecture:

![Microservices Architecture](/screenshots/MicroservicesArchitecture.png)

As shown, the backend is split into several services:
1. **Core**: the remainder of the former monolith, responsible for tenant management, Goods and Production modules (soon to be split into separate services), security, third-party integrations and more.
2. **Supply**, **Demand**, **Notifications**: microservices that handle their respective domains, each with their own MySQL database. There is an additional **Production ML** service responsible for training and running inference on an ML model that optimizes resource distribution in factories.
3. **API Gateway**: Spring Cloud Gateway routing requests to the appropriate services, applying rate limiting, circuit breaking and other cross-cutting concerns.
4. **Config Server**: Spring Cloud Config Server, along with a HashiCorp Vault, providing unified configuration management.
5. **Redis**: used for rate limiting and caching frequently accessed data such as security roles.
6. **Kafka**: used for synchronizing data between services in an event-driven manner, for processing notifications and for broadcasting configuration changes with Spring Cloud Bus.
7. **Prometheus & Grafana** and **ELK Stack** for monitoring and logging.

### How to use
To run the services, follow these steps:
1. Ensure you have installed: JDK21, Maven, Docker and Minikube.
2. Clone this repository and navigate to the root. For every service you wish to use, go to: `cd <service-name>/scripts`. Depending on your OS, run `./BuildImage.ps1` or `./BuildImage.sh` to build the corresponding Docker image. 
3. From the root, go to scripts and run `./RestartMinikube.ps1` or `./RestartMinikube.sh` to start Minikube and deploy the services. *Note that this will remove all existing deployments and services in your Minikube cluster*. 

By default, only the Core and API Gateway services have non-zero replicas. To scale up the others, run `kubectl scale deployment <service-name> --replicas=<number>`. Each Spring Boot service will need a corresponding MySQL instance, named `mysql-<service-name>`. After setting up your cluster, run `minikube tunnel` to expose the services to your local machine and you are ready to use the application.

### Status
In mid-late stages of development.

### Contributing
All contributions are warmly welcomed. Head over to [CONTRIBUTING.md](https://github.com/TudorOrban/ChainOptim-Microservices/blob/main/CONTRIBUTING.md) for details.
