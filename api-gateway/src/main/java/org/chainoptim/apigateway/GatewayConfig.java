package org.chainoptim.apigateway;

import org.springframework.cloud.gateway.route.RouteLocator;
import org.springframework.cloud.gateway.route.builder.RouteLocatorBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class GatewayConfig {

    @Bean
    public RouteLocator customRouteLocator(RouteLocatorBuilder builder) {
        return builder.routes()
            .route("organizations-route", r -> r.path("/api/v1/organizations/**")
                    .uri("http://chainoptim-core:8080"))
            .route("products-route", r -> r.path("/api/v1/products/**")
                .uri("http://chainoptim-core:8080"))
            .route("notifications-route", r -> r.path("/api/v1/notifications/**")
                    .uri("http://chainoptim-notifications:8081"))
            .build();
    }
}
