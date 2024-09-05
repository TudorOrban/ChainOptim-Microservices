package org.chainoptimstorage.shared.sanitization;

import org.chainoptimstorage.core.warehouse.dto.*;
import org.chainoptimstorage.core.warehouseinventoryitem.dto.CreateWarehouseInventoryItemDTO;
import org.chainoptimstorage.core.warehouseinventoryitem.dto.UpdateWarehouseInventoryItemDTO;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class EntitySanitizerServiceImpl implements EntitySanitizerService {

    private final SanitizationService sanitizationService;

    @Autowired
    public EntitySanitizerServiceImpl(SanitizationService sanitizationService) {
        this.sanitizationService = sanitizationService;
    }


    // Warehouses
    public CreateWarehouseDTO sanitizeCreateWarehouseDTO(CreateWarehouseDTO warehouseDTO) {
        warehouseDTO.setName(sanitizationService.sanitize(warehouseDTO.getName()));

        return warehouseDTO;
    }

    public UpdateWarehouseDTO sanitizeUpdateWarehouseDTO(UpdateWarehouseDTO warehouseDTO) {
        warehouseDTO.setName(sanitizationService.sanitize(warehouseDTO.getName()));

        return warehouseDTO;
    }

    public CreateWarehouseInventoryItemDTO sanitizeCreateWarehouseInventoryItemDTO(CreateWarehouseInventoryItemDTO orderDTO) {
        return orderDTO;
    }

    public UpdateWarehouseInventoryItemDTO sanitizeUpdateWarehouseInventoryItemDTO(UpdateWarehouseInventoryItemDTO orderDTO) {
        return orderDTO;
    }


}
