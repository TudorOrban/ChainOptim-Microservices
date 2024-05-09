package org.chainoptimdemand.internal.in.security.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.chainoptimdemand.exception.AuthorizationException;
import org.chainoptimdemand.internal.in.security.dto.SecurityDTO;
import org.chainoptimdemand.internal.in.tenant.model.UserDetailsImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Optional;


@Service("securityService")
public class SecurityServiceImpl implements SecurityService {

    private final HttpClient httpClient = HttpClient.newHttpClient();
    private final ObjectMapper objectMapper;

    @Autowired
    public SecurityServiceImpl(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public boolean canAccessEntity(Long entityId, String entityType, String operationType) {
        return true; // TODO: Call core backend to get authorization
    }

    public boolean canAccessOrganizationEntity(Optional<Integer> organizationId, String entityType, String operationType) {
        System.out.println("Checking if user can access entity in demand");
        String routeAddress = "http://chainoptim-core:8080/api/v1/internal/security/can-access";

        UserDetailsImpl userDetails;
        try {
            userDetails = (UserDetailsImpl) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        } catch (Exception e) {
            throw new AuthorizationException("User not authenticated");
        }
        if (userDetails == null || userDetails.getJwtToken() == null) return false;

        String body;
        try {
            body = objectMapper.writeValueAsString(new SecurityDTO(organizationId.orElse(null), entityType, operationType));
        } catch (Exception e) {
            return false;
        }

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(routeAddress))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + userDetails.getJwtToken())
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(HttpResponse::body)
                .thenApply(Boolean::parseBoolean)
                .join();

    }

    public boolean canUserAccessOrganizationEntity(String userId, String operationType) {
        return true; // TODO: Call core backend to get authorization
    }
}
