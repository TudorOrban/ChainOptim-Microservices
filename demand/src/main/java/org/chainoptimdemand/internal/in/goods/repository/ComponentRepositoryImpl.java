package org.chainoptimdemand.internal.in.goods.repository;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.chainoptimdemand.exception.InternalCommunicationException;
import org.chainoptimdemand.internal.in.goods.model.Component;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Optional;

@Service
public class ComponentRepositoryImpl implements ComponentRepository {

    private final HttpClient httpClient = HttpClient.newHttpClient();
    private final ObjectMapper objectMapper;

    @Autowired
    public ComponentRepositoryImpl(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public Optional<Component> findById(Integer id) {
        String routeAddress = "http://chainoptim-core:8080/api/v1/internal/components/" + id;
        String jwtToken = "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJUdWRvckFPcmJhbjEiLCJvcmdhbml6YXRpb25faWQiOjEsImlhdCI6MTcxNDk3NjQ1MiwiZXhwIjoxNzE1NTgxMjUyfQ.W3Je-xCtcfiazOkEfpoT8bpwy2IDQQG_e8YY1YhT_aG1iWJbxnFnJMtFpWYc036oJD4OmPrefozk_OtI1BAf9g";

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(routeAddress))
                .header("Authorization", "Bearer " + jwtToken)
                .GET()
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(HttpResponse::body)
                .thenApply(response -> {
                    try {
                        return Optional.of(objectMapper.readValue(response, Component.class));
                    } catch (Exception e) {
                        throw new InternalCommunicationException("Failed to fetch component from internal service chainoptim-core");
                    }
                })
                .join();

    }

}
