package org.chainoptim.apigateway;

import org.springframework.cloud.gateway.route.RouteLocator;
import org.springframework.cloud.gateway.route.builder.RouteLocatorBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class GatewayConfig {

    private static final String CORE_SERVICE = "chainoptim-core";
    private static final String NOTIFICATIONS_SERVICE = "chainoptim-notifications";


    @Bean
    public RouteLocator customRouteLocator(RouteLocatorBuilder builder) {
        return builder.routes()
                // Core
                .route("users-route", r -> r.path("/api/v1/users/**")
                        .filters(f -> f.circuitBreaker(c -> c.setName("usersCircuitBreaker")
                                                            .setFallbackUri("forward:/fallback")))
                        .uri("lb://" + CORE_SERVICE))
                .route("login-route", r -> r.path("/api/v1/login/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("validate-jwt-route", r -> r.path("/api/v1/validate-token/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("username-route", r -> r.path("/api/v1/get-username-from-token/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("user-settings-route", r -> r.path("/api/v1/user-settings/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("organizations-route", r -> r.path("/api/v1/organizations/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("organization-invites-route", r -> r.path("/api/v1/organization-invites/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("organization-requests-route", r -> r.path("/api/v1/organization-requests/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("snapshot-route", r -> r.path("/api/v1/supply-chain-snapshots/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("custom-roles-route", r -> r.path("/api/v1/custom-roles/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("products-route", r -> r.path("/api/v1/products/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("stages-route", r -> r.path("/api/v1/stages/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("units-route", r -> r.path("/api/v1/units-of-measurement/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("components-route", r -> r.path("/api/v1/components/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("factories-route", r -> r.path("/api/v1/factories/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("factory-stages-route", r -> r.path("/api/v1/factory-stages/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("warehouses-route", r -> r.path("/api/v1/warehouses/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("suppliers-route", r -> r.path("/api/v1/suppliers/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("supplier-orders-route", r -> r.path("/api/v1/supplier-orders/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("supplier-shipments-route", r -> r.path("/api/v1/supplier-shipments/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("clients-route", r -> r.path("/api/v1/clients/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("client-orders-route", r -> r.path("/api/v1/client-orders/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("client-shipments-route", r -> r.path("/api/v1/client-shipments/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("graphs-route", r -> r.path("/api/v1/graphs/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("product-graphs-route", r -> r.path("/api/v1/product-graphs/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("locations-route", r -> r.path("/api/v1/locations/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("actuator-route", r -> r.path("/api/v1/actuator/prometheus/**")
                        .uri("lb://" + CORE_SERVICE))
                .route("products-route", r -> r.path("/api/v1/products/**")
                        .uri("lb://" + CORE_SERVICE))

                // Internal
                .route("internal-organization-route", r -> r.path("/api/v1/internal/organizations/**")
                        .uri("lb://" + CORE_SERVICE))

                // Notifications
                .route("notifications-route", r -> r.path("/api/v1/notifications/**")
                        .uri("lb://" + NOTIFICATIONS_SERVICE))
                .build();
    }

}
