package org.chainoptimdemand.core.clientshipment.service;

import jakarta.transaction.Transactional;
import org.chainoptimdemand.core.client.dto.CreateClientShipmentDTO;
import org.chainoptimdemand.core.client.dto.ClientDTOMapper;
import org.chainoptimdemand.core.client.dto.UpdateClientShipmentDTO;
import org.chainoptimdemand.exception.PlanLimitReachedException;
import org.chainoptimdemand.exception.ResourceNotFoundException;
import org.chainoptimdemand.exception.ValidationException;
import org.chainoptimdemand.internal.in.tenant.service.SubscriptionPlanLimiterService;
import org.chainoptimdemand.shared.PaginatedResults;
import org.chainoptimdemand.shared.enums.Feature;
import org.chainoptimdemand.internal.in.location.service.LocationService;
import org.chainoptimdemand.shared.sanitization.EntitySanitizerService;
import org.chainoptimdemand.core.clientshipment.model.ClientShipment;
import org.chainoptimdemand.core.clientshipment.repository.ClientShipmentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class ClientShipmentServiceImpl implements ClientShipmentService {

    private final ClientShipmentRepository clientShipmentRepository;
    private final LocationService locationService;
    private final SubscriptionPlanLimiterService planLimiterService;
    private final EntitySanitizerService entitySanitizerService;

    @Autowired
    public ClientShipmentServiceImpl(ClientShipmentRepository clientShipmentRepository,
                                       LocationService locationService,
                                       SubscriptionPlanLimiterService planLimiterService,
                                       EntitySanitizerService entitySanitizerService) {
        this.clientShipmentRepository = clientShipmentRepository;
        this.locationService = locationService;
        this.planLimiterService = planLimiterService;
        this.entitySanitizerService = entitySanitizerService;
    }

    public List<ClientShipment> getClientShipmentsByClientOrderId(Integer orderId) {
        return clientShipmentRepository.findBySupplyOrderId(orderId);
    }

    public List<ClientShipment> getClientShipmentsByClientOrderIds(List<Integer> orderIds) {
        return clientShipmentRepository.findBySupplyOrderIds(orderIds);
    }

    public PaginatedResults<ClientShipment> getClientShipmentsByClientOrderIdAdvanced(Integer clientOrderId, String searchQuery, String sortBy, boolean ascending, int page, int itemsPerPage) {
        return clientShipmentRepository.findByClientOrderIdAdvanced(clientOrderId, searchQuery, sortBy, ascending, page, itemsPerPage);
    }

    public ClientShipment getClientShipmentById(Integer shipmentId) {
        return clientShipmentRepository.findById(shipmentId).orElseThrow(() -> new ResourceNotFoundException("Client shipment with ID: " + shipmentId + " not found."));
    }

    public long countByOrganizationId(Integer organizationId) {
        return clientShipmentRepository.countByOrganizationId(organizationId);
    }

    // Create
    public ClientShipment createClientShipment(CreateClientShipmentDTO shipmentDTO) {
        // Check if plan limit is reached
        if (planLimiterService.isLimitReached(shipmentDTO.getOrganizationId(), Feature.SUPPLIER_SHIPMENT, 1)) {
            throw new PlanLimitReachedException("You have reached the limit of allowed Client Shipments for the current Subscription Plan.");
        }

        // Sanitize input and map to entity
        CreateClientShipmentDTO sanitizedShipmentDTO = entitySanitizerService.sanitizeCreateClientShipmentDTO(shipmentDTO);
        ClientShipment clientShipment = ClientDTOMapper.mapCreateClientShipmentDTOTOShipment(sanitizedShipmentDTO);

//        if (sanitizedShipmentDTO.getSourceLocationId() != null) {
//            Location sourceLocation = locationService.findById(sanitizedShipmentDTO.getSourceLocationId())
//                    .orElseThrow(() -> new ResourceNotFoundException("Source location with ID: " + sanitizedShipmentDTO.getSourceLocationId() + " not found"));
//            clientShipment.setSourceLocation(sourceLocation);
//        }
//
//        if (sanitizedShipmentDTO.getDestinationLocationId() != null) {
//            Location destinationLocation = locationRepository.findById(sanitizedShipmentDTO.getDestinationLocationId())
//                    .orElseThrow(() -> new ResourceNotFoundException("Destination location with ID: " + sanitizedShipmentDTO.getDestinationLocationId() + " not found"));
//            clientShipment.setDestinationLocation(destinationLocation);
//        }

        return clientShipmentRepository.save(clientShipment);
    }

    @Transactional
    public List<ClientShipment> createClientShipmentsInBulk(List<CreateClientShipmentDTO> shipmentDTOs) {
        // Ensure same organizationId
        if (shipmentDTOs.stream().map(CreateClientShipmentDTO::getOrganizationId).distinct().count() > 1) {
            throw new ValidationException("All shipments must belong to the same organization.");
        }
        // Check if plan limit is reached
        if (planLimiterService.isLimitReached(shipmentDTOs.getFirst().getOrganizationId(), Feature.SUPPLIER_SHIPMENT, shipmentDTOs.size())) {
            throw new PlanLimitReachedException("You have reached the limit of allowed Client Shipments for the current Subscription Plan.");
        }

        // Sanitize and map to entity
        List<ClientShipment> shipments = shipmentDTOs.stream()
                .map(shipmentDTO -> {
                    CreateClientShipmentDTO sanitizedShipmentDTO = entitySanitizerService.sanitizeCreateClientShipmentDTO(shipmentDTO);
                    return ClientDTOMapper.mapCreateClientShipmentDTOTOShipment(sanitizedShipmentDTO);
                })
                .toList();

        return clientShipmentRepository.saveAll(shipments);
    }

    @Transactional
    public List<ClientShipment> updateClientShipmentsInBulk(List<UpdateClientShipmentDTO> shipmentDTOs) {
        List<ClientShipment> shipments = new ArrayList<>();
        for (UpdateClientShipmentDTO shipmentDTO : shipmentDTOs) {
            ClientShipment shipment = clientShipmentRepository.findById(shipmentDTO.getId())
                    .orElseThrow(() -> new ResourceNotFoundException("Client Shipment with ID: " + shipmentDTO.getId() + " not found."));

            ClientDTOMapper.setUpdateClientShipmentDTOToClientShipment(shipment, shipmentDTO);
            shipments.add(shipment);
        }

        return clientShipmentRepository.saveAll(shipments);
    }

    @Transactional
    public List<Integer> deleteClientShipmentsInBulk(List<Integer> shipmentIds) {
        List<ClientShipment> shipments = clientShipmentRepository.findAllById(shipmentIds);

        clientShipmentRepository.deleteAll(shipments);

        return shipmentIds;
    }
}
