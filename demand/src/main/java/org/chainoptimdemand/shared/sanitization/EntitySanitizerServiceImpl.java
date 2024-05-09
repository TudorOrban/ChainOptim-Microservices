package org.chainoptimdemand.shared.sanitization;

import org.chainoptimdemand.core.client.dto.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class EntitySanitizerServiceImpl implements EntitySanitizerService {

    private final SanitizationService sanitizationService;

    @Autowired
    public EntitySanitizerServiceImpl(SanitizationService sanitizationService) {
        this.sanitizationService = sanitizationService;
    }


    // Clients
    public CreateClientDTO sanitizeCreateClientDTO(CreateClientDTO clientDTO) {
        clientDTO.setName(sanitizationService.sanitize(clientDTO.getName()));

        return clientDTO;
    }

    public UpdateClientDTO sanitizeUpdateClientDTO(UpdateClientDTO clientDTO) {
        clientDTO.setName(sanitizationService.sanitize(clientDTO.getName()));

        return clientDTO;
    }

    public CreateClientOrderDTO sanitizeCreateClientOrderDTO(CreateClientOrderDTO orderDTO) {
        return orderDTO;
    }

    public UpdateClientOrderDTO sanitizeUpdateClientOrderDTO(UpdateClientOrderDTO orderDTO) {
        return orderDTO;
    }

    public CreateClientShipmentDTO sanitizeCreateClientShipmentDTO(CreateClientShipmentDTO shipmentDTO) {
        return shipmentDTO;
    }

    public UpdateClientShipmentDTO sanitizeUpdateClientShipmentDTO(UpdateClientShipmentDTO shipmentDTO) {
        return shipmentDTO;
    }

}
