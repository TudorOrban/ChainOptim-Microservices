package org.chainoptim.apigateway;

import org.springframework.cloud.gateway.filter.ratelimit.KeyResolver;
import org.springframework.cloud.gateway.filter.ratelimit.RedisRateLimiter;
import org.springframework.cloud.gateway.route.RouteLocator;
import org.springframework.cloud.gateway.route.builder.GatewayFilterSpec;
import org.springframework.cloud.gateway.route.builder.RouteLocatorBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import reactor.core.publisher.Mono;

import java.util.Objects;

@Configuration
public class GatewayConfig {

    private static final String CORE_SERVICE = "lb://chainoptim-core";
    private static final String STORAGE_SERVICE = "lb://chainoptim-storage";
    private static final String SUPPLY_SERVICE = "lb://chainoptim-supply";
    private static final String DEMAND_SERVICE = "lb://chainoptim-demand";
    private static final String NOTIFICATIONS_SERVICE = "lb://chainoptim-notifications";
    private static final String PRODUCTION_ML_SERVICE = "lb://chainoptim-production-ml";

    @Bean
    public RouteLocator customRouteLocator(RouteLocatorBuilder builder, RedisRateLimiter redisRateLimiter) {
        return builder.routes()
                // Core
                .route("users-route", r -> r.path("/api/v1/users/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                .route("auth-route", r -> r.path("/api/v1/auth/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                .route("user-settings-route", r -> r.path("/api/v1/user-settings/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                .route("organizations-route", r -> r.path("/api/v1/organizations/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                .route("organization-invites-route", r -> r.path("/api/v1/organization-invites/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                .route("organization-requests-route", r -> r.path("/api/v1/organization-requests/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                .route("snapshot-route", r -> r.path("/api/v1/supply-chain-snapshots/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                .route("custom-roles-route", r -> r.path("/api/v1/custom-roles/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                .route("products-route", r -> r.path("/api/v1/products/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                .route("stages-route", r -> r.path("/api/v1/stages/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                .route("units-route", r -> r.path("/api/v1/units-of-measurement/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                .route("components-route", r -> r.path("/api/v1/components/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                .route("factories-route", r -> r.path("/api/v1/factories/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                .route("factory-stages-route", r -> r.path("/api/v1/factory-stages/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                .route("factory-production-histories-route", r -> r.path("/api/v1/factory-production-histories/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                .route("factory-graphs-route", r -> r.path("/api/v1/factory-graphs/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                .route("product-graphs-route", r -> r.path("/api/v1/product-graphs/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                .route("locations-route", r -> r.path("/api/v1/locations/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))

                // Storage
                .route("warehouses-route", r -> r.path("/api/v1/warehouses/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(STORAGE_SERVICE))
                .route("warehouse-inventory-route, ", r -> r.path("/api/v1/warehouse-inventory-items/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(STORAGE_SERVICE))
                .route("compartments-route", r -> r.path("/api/v1/compartments/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(STORAGE_SERVICE))
                .route("crates-route", r -> r.path("/api/v1/crates/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(STORAGE_SERVICE))

                // Supply
                .route("suppliers-route", r -> r.path("/api/v1/suppliers/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(SUPPLY_SERVICE))
                .route("supplier-orders-route", r -> r.path("/api/v1/supplier-orders/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(SUPPLY_SERVICE))
                .route("supplier-shipments-route", r -> r.path("/api/v1/supplier-shipments/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(SUPPLY_SERVICE))
                .route("supplier-performance-route", r -> r.path("/api/v1/supplier-performance/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(SUPPLY_SERVICE))

                // Demand
                .route("clients-route", r -> r.path("/api/v1/clients/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(DEMAND_SERVICE))
                .route("client-orders-route", r -> r.path("/api/v1/client-orders/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(DEMAND_SERVICE))
                .route("client-shipments-route", r -> r.path("/api/v1/client-shipments/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(DEMAND_SERVICE))
                .route("actuator-route", r -> r.path("/api/v1/actuator/prometheus/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))
                // Notifications
                .route("notifications-route", r -> r.path("/api/v1/notifications/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(NOTIFICATIONS_SERVICE))
                .route("websocket-route", r -> r.path("/ws/**")
                        .uri(NOTIFICATIONS_SERVICE))
                // ML
                .route("production-ml-route", r -> r.path("/api/v1/ml/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(PRODUCTION_ML_SERVICE))
                // Internal
                .route("internal-organization-route", r -> r.path("/api/v1/internal/organizations/**")
                        .filters(f -> customFilter(f, redisRateLimiter))
                        .uri(CORE_SERVICE))

                .build();
    }

    private GatewayFilterSpec customFilter(GatewayFilterSpec builder, RedisRateLimiter redisRateLimiter) {
        return builder.requestRateLimiter(c -> {
                        c.setRateLimiter(redisRateLimiter);
                        c.setKeyResolver(ipKeyResolver());
                    })
                    .circuitBreaker(c -> c.setName("default").setFallbackUri("forward:/fallback"));
    }

    @Bean
    public RedisRateLimiter redisRateLimiter() {
        return new RedisRateLimiter(10, 10);
    }

    @Bean
    public KeyResolver ipKeyResolver() {
        return exchange -> Mono.just(Objects.requireNonNull(exchange.getRequest().getRemoteAddress()).toString());
    }


}
