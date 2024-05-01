package org.chainoptimnotifications.user.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.chainoptimnotifications.user.model.Organization;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

@Service
public class OrganizationRepositoryImpl implements OrganizationRepository {

    private final HttpClient httpClient = HttpClient.newHttpClient();

    public Organization getOrganizationWithUsersAndCustomRoles(Integer id) {
        String routeAddress = "http://localhost:8080/api/v1/organizations-internal/" + id;
        String jwtToken = "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJUdWRvckFPcmJhbjEiLCJvcmdhbml6YXRpb25faWQiOjEsImlhdCI6MTcxNDMzOTg5MCwiZXhwIjoxNzE0OTQ0NjkwfQ.sR98XrH6oKjFSU_-FevFIk-Cp_UqgDyaa8bmJqn7SCW0s6TH1PZyynGIqYiyeUfm0qOCcYuFU9Cd-RiD2NN6Lg";

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(routeAddress))
                .headers("Authorization", "Bearer " + jwtToken)
                .GET()
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(HttpResponse::body)
                .thenApply(response -> {
                    System.out.println(response);
                    try {
                        return new ObjectMapper().readValue(response, Organization.class);
                    } catch (Exception e) {
                        e.printStackTrace();
                        return null;
                    }
                })
                .join();

    }
}
