package org.chainoptim.shared.commonfeatures.location.service;

import org.chainoptim.shared.commonfeatures.location.model.Location;
import org.chainoptim.shared.commonfeatures.location.repository.LocationRepository;
import io.grpc.stub.StreamObserver;
import location.LocationServiceGrpc;
import net.devh.boot.grpc.server.service.GrpcService;
import org.springframework.security.access.prepost.PreAuthorize;

import java.util.List;

@GrpcService
public class GrpcLocationService extends LocationServiceGrpc.LocationServiceImplBase {

    private final LocationRepository locationRepository;

    public GrpcLocationService(LocationRepository locationRepository) {
        this.locationRepository = locationRepository;
    }

    @Override
    @PreAuthorize("@securityService.canAccessOrganizationEntity(#request.organizationId, 'Location', 'Read')")
    public void getLocationsByOrganizationId(location.Location.OrganizationIdRequest request, StreamObserver<location.Location.LocationResponse> responseObserver) {
        try {
            List<Location> locations = locationRepository.findLocationsByOrganizationId(request.getOrganizationId());
            location.Location.LocationResponse.Builder responseBuilder = location.Location.LocationResponse.newBuilder();
            for (Location entityLocation : locations) {
                location.Location.GrpcLocation protoLocation = location.Location.GrpcLocation.newBuilder()
                        .setId(entityLocation.getId())
                        .setAddress(entityLocation.getAddress())
                        .setCity(entityLocation.getCity())
                        .setState(entityLocation.getState())
                        .setCountry(entityLocation.getCountry())
                        .setZipCode(entityLocation.getZipCode())
                        .setLatitude(entityLocation.getLatitude())
                        .setLongitude(entityLocation.getLongitude())
                        .build();
                responseBuilder.addLocations(protoLocation);
            }
            responseObserver.onNext(responseBuilder.build());
            responseObserver.onCompleted();
        } catch (Exception e) {
            responseObserver.onError(io.grpc.Status.INTERNAL.withDescription(e.getMessage()).asRuntimeException());
        }
    }
}
