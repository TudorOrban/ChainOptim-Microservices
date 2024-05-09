package org.chainoptimdemand.core.clientshipment.service;


import org.chainoptimdemand.core.client.dto.CreateClientShipmentDTO;
import org.chainoptimdemand.core.client.dto.UpdateClientShipmentDTO;
import org.chainoptimdemand.shared.PaginatedResults;
import org.chainoptimdemand.core.clientshipment.model.ClientShipment;

import java.util.List;

public interface ClientShipmentService {

    List<ClientShipment> getClientShipmentsByClientOrderId(Integer orderId);
    List<ClientShipment> getClientShipmentsByClientOrderIds(List<Integer> orderIds);
    PaginatedResults<ClientShipment> getClientShipmentsByClientOrderIdAdvanced(Integer clientOrderId, String searchQuery, String sortBy, boolean ascending, int page, int itemsPerPage);
    ClientShipment getClientShipmentById(Integer shipmentId);
    long countByOrganizationId(Integer organizationId);
    ClientShipment createClientShipment(CreateClientShipmentDTO shipmentDTO);
    List<ClientShipment> createClientShipmentsInBulk(List<CreateClientShipmentDTO> shipmentDTOs);
    List<ClientShipment> updateClientShipmentsInBulk(List<UpdateClientShipmentDTO> shipmentDTOs);
    List<Integer> deleteClientShipmentsInBulk(List<Integer> shipmentIds);
}
