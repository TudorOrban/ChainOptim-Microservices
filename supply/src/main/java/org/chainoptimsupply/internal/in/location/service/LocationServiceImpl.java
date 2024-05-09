package org.chainoptimsupply.internal.in.location.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.chainoptimsupply.internal.in.location.dto.CreateLocationDTO;
import org.chainoptimsupply.internal.in.location.dto.Location;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

@Service
public class LocationServiceImpl implements LocationService {

    private final HttpClient httpClient = HttpClient.newHttpClient();
    private final ObjectMapper objectMapper;

    @Autowired
    public LocationServiceImpl(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public Location createLocation(CreateLocationDTO locationDTO) {
        String routeAddress = "http://chainoptim-core:8080/api/v1/internal/locations/create";
        String jwtToken = "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJUdWRvckFPcmJhbjEiLCJvcmdhbml6YXRpb25faWQiOjEsImlhdCI6MTcxNDk3NjQ1MiwiZXhwIjoxNzE1NTgxMjUyfQ.W3Je-xCtcfiazOkEfpoT8bpwy2IDQQG_e8YY1YhT_aG1iWJbxnFnJMtFpWYc036oJD4OmPrefozk_OtI1BAf9g";

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(routeAddress))
                .header("Authorization", "Bearer " + jwtToken)
                .POST(HttpRequest.BodyPublishers.ofString(locationDTO.toString()))
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(HttpResponse::body)
                .thenApply(response -> {
                    System.out.println(response);
                    try {
                        return objectMapper.readValue(response, new TypeReference<Location>() {});
                    } catch (Exception e) {
                        e.printStackTrace();
                        return null;
                    }
                })
                .join();

    }

}