package org.chainoptimstorage.shared.sanitization;

import org.chainoptimstorage.core.warehouse.dto.*;

public interface EntitySanitizerService {

    // Warehouse
    CreateWarehouseDTO sanitizeCreateWarehouseDTO(CreateWarehouseDTO warehouseDTO);
    UpdateWarehouseDTO sanitizeUpdateWarehouseDTO(UpdateWarehouseDTO warehouseDTO);
//    CreateWarehouseOrderDTO sanitizeCreateWarehouseOrderDTO(CreateWarehouseOrderDTO orderDTO);
//    UpdateWarehouseOrderDTO sanitizeUpdateWarehouseOrderDTO(UpdateWarehouseOrderDTO orderDTO);
//    CreateWarehouseShipmentDTO sanitizeCreateWarehouseShipmentDTO(CreateWarehouseShipmentDTO shipmentDTO);
//    UpdateWarehouseShipmentDTO sanitizeUpdateWarehouseShipmentDTO(UpdateWarehouseShipmentDTO shipmentDTO);

}
