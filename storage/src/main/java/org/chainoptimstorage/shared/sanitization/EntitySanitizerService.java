package org.chainoptimstorage.shared.sanitization;

import org.chainoptimstorage.core.warehouse.dto.*;
import org.chainoptimstorage.core.warehouseinventoryitem.dto.CreateWarehouseInventoryItemDTO;
import org.chainoptimstorage.core.warehouseinventoryitem.dto.UpdateWarehouseInventoryItemDTO;

public interface EntitySanitizerService {

    // Warehouse
    CreateWarehouseDTO sanitizeCreateWarehouseDTO(CreateWarehouseDTO warehouseDTO);
    UpdateWarehouseDTO sanitizeUpdateWarehouseDTO(UpdateWarehouseDTO warehouseDTO);
    CreateWarehouseInventoryItemDTO sanitizeCreateWarehouseInventoryItemDTO(CreateWarehouseInventoryItemDTO itemDTO);
    UpdateWarehouseInventoryItemDTO sanitizeUpdateWarehouseInventoryItemDTO(UpdateWarehouseInventoryItemDTO itemDTO);

}
