package org.chainoptimdemand.shared.sanitization;

import org.chainoptimdemand.core.client.dto.*;

public interface EntitySanitizerService {

    // Client
    CreateClientDTO sanitizeCreateClientDTO(CreateClientDTO clientDTO);
    UpdateClientDTO sanitizeUpdateClientDTO(UpdateClientDTO clientDTO);
    CreateClientOrderDTO sanitizeCreateClientOrderDTO(CreateClientOrderDTO orderDTO);
    UpdateClientOrderDTO sanitizeUpdateClientOrderDTO(UpdateClientOrderDTO orderDTO);
    CreateClientShipmentDTO sanitizeCreateClientShipmentDTO(CreateClientShipmentDTO shipmentDTO);
    UpdateClientShipmentDTO sanitizeUpdateClientShipmentDTO(UpdateClientShipmentDTO shipmentDTO);

}
