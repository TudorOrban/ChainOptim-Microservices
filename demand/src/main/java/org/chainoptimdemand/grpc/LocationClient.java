package org.chainoptimdemand.grpc;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.chainoptimdemand.internal.in.location.dto.Location;

import java.util.List;

public class LocationClient {

    private final location.LocationServiceGrpc.LocationServiceBlockingStub locationServiceBlockingStub;

    public LocationClient() {
        String host = System.getenv("GRPC_SERVICE_HOST");
        int port = Integer.parseInt(System.getenv("GRPC_SERVICE_PORT"));
        ManagedChannel channel = ManagedChannelBuilder.forAddress(host, port)
                .usePlaintext()
                .build();
        locationServiceBlockingStub = location.LocationServiceGrpc.newBlockingStub(channel);
    }

    public List<Location> getLocationsByOrganizationId(Integer organizationId) {
        location.Location.OrganizationIdRequest request = location.Location.OrganizationIdRequest.newBuilder()
                .setOrganizationId(organizationId)
                .build();
        location.Location.LocationResponse response = locationServiceBlockingStub.getLocationsByOrganizationId(request);
        return response.getLocationsList().stream()
                .map(grpcLocation -> new Location(
                        grpcLocation.getId(),
                        grpcLocation.getAddress(),
                        grpcLocation.getCity(),
                        grpcLocation.getState(),
                        grpcLocation.getCountry(),
                        grpcLocation.getZipCode(),
                        grpcLocation.getLatitude(),
                        grpcLocation.getLongitude(),
                        grpcLocation.getOrganizationId()))
                .toList();
    }
}
