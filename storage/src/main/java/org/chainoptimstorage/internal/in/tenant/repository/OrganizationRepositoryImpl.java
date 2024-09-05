package org.chainoptimstorage.internal.in.tenant.repository;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.chainoptimstorage.internal.in.tenant.model.Organization;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

@Service
public class OrganizationRepositoryImpl implements OrganizationRepository {

    private final HttpClient httpClient = HttpClient.newHttpClient();
    private final ObjectMapper objectMapper;

    @Autowired
    public OrganizationRepositoryImpl(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public Organization getOrganizationWithUsersAndCustomRoles(Integer id) {
        String routeAddress = "http://chainoptim-core:8080/api/v1/internal/organizations/" + id;
        String jwtToken = "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJUdWRvckFPcmJhbjEiLCJvcmdhbml6YXRpb25faWQiOjEsImlhdCI6MTcxNDk3NjQ1MiwiZXhwIjoxNzE1NTgxMjUyfQ.W3Je-xCtcfiazOkEfpoT8bpwy2IDQQG_e8YY1YhT_aG1iWJbxnFnJMtFpWYc036oJD4OmPrefozk_OtI1BAf9g";

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(routeAddress))
                .header("Authorization", "Bearer " + jwtToken)
                .GET()
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(HttpResponse::body)
                .thenApply(response -> {
                    System.out.println(response);
                    try {
                        return objectMapper.readValue(response, Organization.class);
                    } catch (Exception e) {
                        e.printStackTrace();
                        return null;
                    }
                })
                .join();

    }

    public Organization.SubscriptionPlanTier getSubscriptionPlanTierById(Integer id) {
        String routeAddress = "http://chainoptim-core:8080/api/v1/internal/organizations/" + id + "/subscription-plan-tier";
        String jwtToken = "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJUdWRvckFPcmJhbjEiLCJvcmdhbml6YXRpb25faWQiOjEsImlhdCI6MTcxNDk3NjQ1MiwiZXhwIjoxNzE1NTgxMjUyfQ.W3Je-xCtcfiazOkEfpoT8bpwy2IDQQG_e8YY1YhT_aG1iWJbxnFnJMtFpWYc036oJD4OmPrefozk_OtI1BAf9g";

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(routeAddress))
                .header("Authorization", "Bearer " + jwtToken)
                .GET()
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(HttpResponse::body)
                .thenApply(response -> {
                    System.out.println(response);
                    try {
                        return objectMapper.readValue(response, Organization.SubscriptionPlanTier.class);
                    } catch (Exception e) {
                        e.printStackTrace();
                        return null;
                    }
                })
                .join();

    }
}
