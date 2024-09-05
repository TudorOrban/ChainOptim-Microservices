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
}
