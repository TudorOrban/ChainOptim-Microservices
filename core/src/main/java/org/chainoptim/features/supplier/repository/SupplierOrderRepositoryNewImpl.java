package org.chainoptim.features.supplier.repository;

import org.chainoptim.exception.ValidationException;
import org.chainoptim.features.supplier.model.SupplierOrder;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;
import java.util.Optional;

@Service
public class SupplierOrderRepositoryNewImpl implements SupplierOrderRepositoryNew {

    private static final Logger logger = LoggerFactory.getLogger(SupplierOrderRepositoryNewImpl.class);
    private final HttpClient httpClient = HttpClient.newHttpClient();

    public List<SupplierOrder> findSupplierOrdersByOrganizationId(Integer organizationId) {
        logger.info("Attempting to load supplier orders by organizationId: {}", organizationId);

        String routeAddress = "http://chainoptim-core/api/v1/internal/supplier-orders/organization/" + organizationId;

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(routeAddress))
                .GET()
                .build();

        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return new ObjectMapper().readValue(response.body(), new TypeReference<List<SupplierOrder>>() {});
        } catch (Exception e) {
            logger.error("Error occurred while fetching supplier orders by organizationId: {}", organizationId);
            throw new ValidationException("Supplier orders malformed.");
        }
    }

    public Optional<Integer> findOrganizationIdById(Long supplierOrderId) {
        String routeAddress = "http://chainoptim-core/api/v1/internal/supplier-orders/" + supplierOrderId + "/organization-id";

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(routeAddress))
                .GET()
                .build();

        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return Optional.of(Integer.parseInt(response.body()));
        } catch (Exception e) {
            logger.error("Error occurred while fetching organizationId by supplierOrderId: {}", supplierOrderId);
            throw new ValidationException("OrganizationId malformed.");
        }
    }

    public long countByOrganizationId(@Param("organizationId") Integer organizationId) {
        String routeAddress = "http://chainoptim-core/api/v1/internal/supplier-orders/organization/" + organizationId + "/count";

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(routeAddress))
                .GET()
                .build();

        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return Long.parseLong(response.body());
        } catch (Exception e) {
            logger.error("Error occurred while fetching count by organizationId: {}", organizationId);
            throw new ValidationException("Count malformed.");
        }
    }
}
