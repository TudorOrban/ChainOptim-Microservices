package org.chainoptim.config;

import io.grpc.Metadata;
import io.grpc.ServerCall;
import net.devh.boot.grpc.server.security.authentication.GrpcAuthenticationReader;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;


import javax.naming.AuthenticationException;

@Configuration
public class GrpcSecurityConfig {

    @Bean
    public GrpcAuthenticationReader grpcAuthenticationReader() {
        return new BasicGrpcAuthenticationReader(); // or use your custom implementation
    }

    // Basic implementation example
    static class BasicGrpcAuthenticationReader implements GrpcAuthenticationReader {
        @Override
        public Authentication readAuthentication(ServerCall<?, ?> call, Metadata headers) {
            // Your logic to extract authentication data from the call and headers
            return new UsernamePasswordAuthenticationToken("user", "password"); // Example
        }
    }
}