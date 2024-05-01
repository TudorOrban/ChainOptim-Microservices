package org.chainoptimnotifications.user.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.chainoptimnotifications.user.model.User;
import org.chainoptimnotifications.user.model.UserDetailsImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Collections;

@Service
public class AuthenticationServiceImpl implements AuthenticationService {

    private static final Logger logger = LoggerFactory.getLogger(AuthenticationServiceImpl.class);
    private final HttpClient httpClient = HttpClient.newHttpClient();

    public UserDetailsImpl loadUserByUsername(String username) {
        logger.info("Attempting to load user by username: {}", username);

        String routeAddress = "http://localhost:8080/api/v1/users/" + username;

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(routeAddress))
                .GET()
                .build();

        User user = null;
        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            user = new ObjectMapper().readValue(response.body(), User.class);
        } catch (Exception e) {
            logger.error("Error occurred while fetching user by username: {}", username);
            throw new UsernameNotFoundException("User not found");
        }

        UserDetailsImpl userDetails = new UserDetailsImpl();
        userDetails.setUsername(user.getUsername());
        userDetails.setPassword(user.getPasswordHash());
        userDetails.setOrganizationId(user.getOrganization().getId());
        userDetails.setRole(user.getRole());
        userDetails.setCustomRole(user.getCustomRole());
        userDetails.setAuthorities(Collections.singletonList(new SimpleGrantedAuthority("ROLE_USER")));

        return userDetails;
    }
}
