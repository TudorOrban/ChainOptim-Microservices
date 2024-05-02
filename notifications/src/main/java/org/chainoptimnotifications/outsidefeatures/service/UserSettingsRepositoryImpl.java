package org.chainoptimnotifications.outsidefeatures.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.chainoptimnotifications.outsidefeatures.model.UserSettings;
import org.chainoptimnotifications.user.model.Organization;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.util.UriComponentsBuilder;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.List;

@Service
public class UserSettingsRepositoryImpl implements UserSettingsRepository {

    private final HttpClient httpClient = HttpClient.newHttpClient();
    private final ObjectMapper objectMapper;

    @Autowired
    public UserSettingsRepositoryImpl(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public List<UserSettings> getSettingsByUserIds(List<String> userIds) {
        if (userIds.isEmpty()) return new ArrayList<>();

        String routeAddress = "http://chainoptim-core:8080/api/v1/internal/settings/users";
        URI uri = UriComponentsBuilder.fromHttpUrl(routeAddress)
                .queryParam("userIds", String.join(",", userIds))  // Join user IDs into a single query parameter
                .build()
                .toUri();

        String jwtToken = "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJUdWRvckFPcmJhbjEiLCJvcmdhbml6YXRpb25faWQiOjEsImlhdCI6MTcxNDMzOTg5MCwiZXhwIjoxNzE0OTQ0NjkwfQ.sR98XrH6oKjFSU_-FevFIk-Cp_UqgDyaa8bmJqn7SCW0s6TH1PZyynGIqYiyeUfm0qOCcYuFU9Cd-RiD2NN6Lg";

        HttpRequest request = HttpRequest.newBuilder()
                .uri(uri)
                .header("Authorization", "Bearer " + jwtToken)
                .GET()
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(HttpResponse::body)
                .thenApply(response -> {
                    System.out.println(response);
                    try {
                        return objectMapper.readValue(response, new TypeReference<List<UserSettings>>() {});
                    } catch (Exception e) {
                        e.printStackTrace();
                        return null;
                    }
                })
                .join();

    }
}
