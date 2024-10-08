package org.chainoptimnotifications.internal.in.settings.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.chainoptimnotifications.internal.in.settings.model.UserSettings;
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

        String jwtToken = "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJUdWRvckFPcmJhbjEiLCJvcmdhbml6YXRpb25faWQiOjEsImlhdCI6MTcxNDk3NjQ1MiwiZXhwIjoxNzE1NTgxMjUyfQ.W3Je-xCtcfiazOkEfpoT8bpwy2IDQQG_e8YY1YhT_aG1iWJbxnFnJMtFpWYc036oJD4OmPrefozk_OtI1BAf9g";

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
