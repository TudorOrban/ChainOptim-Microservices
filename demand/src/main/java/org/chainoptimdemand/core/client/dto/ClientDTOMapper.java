package org.chainoptimdemand.core.client.dto;

import org.chainoptimdemand.core.clientorder.model.ClientOrder;
import org.chainoptimdemand.core.client.model.Client;
import org.chainoptimdemand.core.clientshipment.model.ClientShipment;

public class ClientDTOMapper {

    private ClientDTOMapper() {}

    public static ClientsSearchDTO convertToClientsSearchDTO(Client client) {
        ClientsSearchDTO dto = new ClientsSearchDTO();
        dto.setId(client.getId());
        dto.setName(client.getName());
        dto.setCreatedAt(client.getCreatedAt());
        dto.setUpdatedAt(client.getUpdatedAt());
        dto.setLocationId(client.getLocationId());
        return dto;
    }

    public static Client convertCreateClientDTOToClient(CreateClientDTO clientDTO) {
        Client client = new Client();
        client.setName(clientDTO.getName());
        client.setOrganizationId(clientDTO.getOrganizationId());
        if (clientDTO.getLocationId() != null) {
            client.setLocationId(clientDTO.getLocationId());
        }

        return client;
    }

    public static ClientOrder mapCreateDtoToClientOrder(CreateClientOrderDTO order) {
        ClientOrder clientOrder = new ClientOrder();
        clientOrder.setOrganizationId(order.getOrganizationId());
        clientOrder.setClientId(order.getClientId());
        clientOrder.setQuantity(order.getQuantity());
        clientOrder.setDeliveredQuantity(order.getDeliveredQuantity());
        clientOrder.setOrderDate(order.getOrderDate());
        clientOrder.setEstimatedDeliveryDate(order.getEstimatedDeliveryDate());
        clientOrder.setDeliveryDate(order.getDeliveryDate());
        clientOrder.setStatus(order.getStatus());
        clientOrder.setCompanyId(order.getCompanyId());

        return clientOrder;
    }

    public static void setUpdateClientOrderDTOToUpdateOrder(ClientOrder clientOrder, UpdateClientOrderDTO orderDTO) {
        clientOrder.setQuantity(orderDTO.getQuantity());
        clientOrder.setDeliveredQuantity(orderDTO.getDeliveredQuantity());
        clientOrder.setOrderDate(orderDTO.getOrderDate());
        clientOrder.setEstimatedDeliveryDate(orderDTO.getEstimatedDeliveryDate());
        clientOrder.setDeliveryDate(orderDTO.getDeliveryDate());
        clientOrder.setStatus(orderDTO.getStatus());
        clientOrder.setCompanyId(orderDTO.getCompanyId());
    }

    public static ClientShipment mapCreateClientShipmentDTOTOShipment(CreateClientShipmentDTO shipmentDTO) {
        ClientShipment shipment = new ClientShipment();
        shipment.setClientOrderId(shipmentDTO.getClientOrderId());
        shipment.setQuantity(shipmentDTO.getQuantity());
        shipment.setShipmentStartingDate(shipmentDTO.getShipmentStartingDate());
        shipment.setEstimatedArrivalDate(shipmentDTO.getEstimatedArrivalDate());
        shipment.setArrivalDate(shipmentDTO.getArrivalDate());
        shipment.setStatus(shipmentDTO.getStatus());
        shipment.setCurrentLocationLatitude(shipmentDTO.getCurrentLocationLatitude());
        shipment.setCurrentLocationLongitude(shipmentDTO.getCurrentLocationLongitude());

        return shipment;
    }

    public static void setUpdateClientShipmentDTOToClientShipment(ClientShipment shipment, UpdateClientShipmentDTO shipmentDTO) {
        shipment.setQuantity(shipmentDTO.getQuantity());
        shipment.setShipmentStartingDate(shipmentDTO.getShipmentStartingDate());
        shipment.setEstimatedArrivalDate(shipmentDTO.getEstimatedArrivalDate());
        shipment.setArrivalDate(shipmentDTO.getArrivalDate());
        shipment.setStatus(shipmentDTO.getStatus());
        shipment.setCurrentLocationLatitude(shipmentDTO.getCurrentLocationLatitude());
        shipment.setCurrentLocationLongitude(shipmentDTO.getCurrentLocationLongitude());
    }
}
