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
                        .uri("lb://chainoptim-core"))
                .route("notifications-route", r -> r.path("/api/v1/notification/**")
                        .uri("lb://chainoptim-notifications"))
                .build();
    }
}
