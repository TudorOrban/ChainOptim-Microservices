package org.chainoptimstorage.core.warehouse.dto;

import org.chainoptimstorage.core.warehouse.model.Warehouse;
import org.chainoptimstorage.internal.in.location.dto.Location;

public class WarehouseDTOMapper {

    private WarehouseDTOMapper() {}

    public static WarehousesSearchDTO convertToWarehousesSearchDTO(Warehouse warehouse) {
        WarehousesSearchDTO dto = new WarehousesSearchDTO();
        dto.setId(warehouse.getId());
        dto.setName(warehouse.getName());
        dto.setCreatedAt(warehouse.getCreatedAt());
        dto.setUpdatedAt(warehouse.getUpdatedAt());
        dto.setLocation(warehouse.getLocation());
        return dto;
    }
    
    public static Warehouse mapCreateWarehouseDTOToWarehouse(CreateWarehouseDTO warehouseDTO) {
        Warehouse warehouse = new Warehouse();
        warehouse.setName(warehouseDTO.getName());
        warehouse.setOrganizationId(warehouseDTO.getOrganizationId());
        if (warehouseDTO.getLocationId() != null) {
            Location location = new Location();
            location.setId(warehouseDTO.getLocationId());
            warehouse.setLocation(location);
        }

        return warehouse;
    }
//
//    public static WarehouseOrder mapCreateDtoToWarehouseOrder(CreateWarehouseOrderDTO order) {
//        WarehouseOrder warehouseOrder = new WarehouseOrder();
//        warehouseOrder.setOrganizationId(order.getOrganizationId());
//        warehouseOrder.setWarehouseId(order.getWarehouseId());
//        warehouseOrder.setQuantity(order.getQuantity());
//        warehouseOrder.setDeliveredQuantity(order.getDeliveredQuantity());
//        warehouseOrder.setOrderDate(order.getOrderDate());
//        warehouseOrder.setEstimatedDeliveryDate(order.getEstimatedDeliveryDate());
//        warehouseOrder.setDeliveryDate(order.getDeliveryDate());
//        warehouseOrder.setStatus(order.getStatus());
//        warehouseOrder.setCompanyId(order.getCompanyId());
//
//        return warehouseOrder;
//    }
//
//    public static void setUpdateWarehouseOrderDTOToUpdateOrder(WarehouseOrder warehouseOrder, UpdateWarehouseOrderDTO orderDTO) {
//        warehouseOrder.setQuantity(orderDTO.getQuantity());
//        warehouseOrder.setDeliveredQuantity(orderDTO.getDeliveredQuantity());
//        warehouseOrder.setOrderDate(orderDTO.getOrderDate());
//        warehouseOrder.setEstimatedDeliveryDate(orderDTO.getEstimatedDeliveryDate());
//        warehouseOrder.setDeliveryDate(orderDTO.getDeliveryDate());
//        warehouseOrder.setStatus(orderDTO.getStatus());
//        warehouseOrder.setCompanyId(orderDTO.getCompanyId());
//    }
//
//    public static WarehouseShipment mapCreateWarehouseShipmentDTOTOShipment(CreateWarehouseShipmentDTO shipmentDTO) {
//        WarehouseShipment shipment = new WarehouseShipment();
//        shipment.setWarehouseOrderId(shipmentDTO.getWarehouseOrderId());
//        shipment.setQuantity(shipmentDTO.getQuantity());
//        shipment.setShipmentStartingDate(shipmentDTO.getShipmentStartingDate());
//        shipment.setEstimatedArrivalDate(shipmentDTO.getEstimatedArrivalDate());
//        shipment.setArrivalDate(shipmentDTO.getArrivalDate());
//        shipment.setStatus(shipmentDTO.getStatus());
//        shipment.setCurrentLocationLatitude(shipmentDTO.getCurrentLocationLatitude());
//        shipment.setCurrentLocationLongitude(shipmentDTO.getCurrentLocationLongitude());
//
//        return shipment;
//    }
//
//    public static void setUpdateWarehouseShipmentDTOToWarehouseShipment(WarehouseShipment shipment, UpdateWarehouseShipmentDTO shipmentDTO) {
//        shipment.setQuantity(shipmentDTO.getQuantity());
//        shipment.setShipmentStartingDate(shipmentDTO.getShipmentStartingDate());
//        shipment.setEstimatedArrivalDate(shipmentDTO.getEstimatedArrivalDate());
//        shipment.setArrivalDate(shipmentDTO.getArrivalDate());
//        shipment.setStatus(shipmentDTO.getStatus());
//        shipment.setCurrentLocationLatitude(shipmentDTO.getCurrentLocationLatitude());
//        shipment.setCurrentLocationLongitude(shipmentDTO.getCurrentLocationLongitude());
//    }
}
