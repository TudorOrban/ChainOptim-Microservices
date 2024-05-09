package org.chainoptim.internalcommunication.in.supplier.repository;

import org.chainoptim.exception.ValidationException;
import org.chainoptim.internalcommunication.in.supplier.model.SupplierShipment;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;

@Service
public class SupplierShipmentRepositoryImpl implements SupplierShipmentRepository {

    private static final Logger logger = LoggerFactory.getLogger(SupplierShipmentRepositoryImpl.class);
    private final HttpClient httpClient = HttpClient.newHttpClient();

    public List<SupplierShipment> findSupplierShipmentsBySupplierOrderIds(List<Integer> orderIds) {
        logger.info("Attempting to load supplier shipments by orderIds: {}", orderIds);

        String routeAddress = "http://chainoptim-core/api/v1/internal/supplier-shipments/supplier-orders/" + orderIds;

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(routeAddress))
                .GET()
                .build();

        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return new ObjectMapper().readValue(response.body(), new TypeReference<List<SupplierShipment>>() {});
        } catch (Exception e) {
            logger.error("Error occurred while fetching supplier shipments by supplierOrderIds: {}", orderIds);
            throw new ValidationException("Supplier orders malformed.");
        }
    }

    public long countByOrganizationId(Integer organizationId) {
        logger.info("Attempting to count supplier shipments by organizationId: {}", organizationId);

        String routeAddress = "http://chainoptim-core/api/v1/internal/supplier-shipments/organization/" + organizationId + "/count";

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(routeAddress))
                .GET()
                .build();

        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return Long.parseLong(response.body());
        } catch (Exception e) {
            logger.error("Error occurred while counting supplier shipments by organizationId: {}", organizationId);
            throw new ValidationException("Supplier shipments count malformed.");
        }
    }
}
