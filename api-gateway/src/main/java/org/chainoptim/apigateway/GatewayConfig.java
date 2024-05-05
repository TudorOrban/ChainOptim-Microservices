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
                        .uri("lb://chainoptim-core"))  // Use load-balanced URI
                .route("products-route", r -> r.path("/api/v1/products/**")
                        .uri("lb://chainoptim-core"))  // Use load-balanced URI
                .route("notifications-route", r -> r.path("/api/v1/notifications/**")
                        .uri("lb://chainoptim-notifications"))  // Use load-balanced URI
                .build();
    }

}
