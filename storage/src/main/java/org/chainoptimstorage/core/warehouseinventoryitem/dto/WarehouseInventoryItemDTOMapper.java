package org.chainoptimstorage.core.warehouseinventoryitem.dto;

import org.chainoptimstorage.core.warehouseinventoryitem.model.WarehouseInventoryItem;

public class WarehouseInventoryItemDTOMapper {

    private WarehouseInventoryItemDTOMapper() {}

    public static WarehouseInventoryItem mapCreateDtoToWarehouseInventoryItem(CreateWarehouseInventoryItemDTO item) {
        WarehouseInventoryItem warehouseInventoryItem = new WarehouseInventoryItem();
        warehouseInventoryItem.setOrganizationId(item.getOrganizationId());
        warehouseInventoryItem.setWarehouseId(item.getWarehouseId());
        warehouseInventoryItem.setQuantity(item.getQuantity());
        warehouseInventoryItem.setCompanyId(item.getCompanyId());

        return warehouseInventoryItem;
    }

    public static void setUpdateWarehouseInventoryItemDTOToUpdateInventoryItem(WarehouseInventoryItem warehouseInventoryItem, UpdateWarehouseInventoryItemDTO itemDTO) {
        warehouseInventoryItem.setQuantity(itemDTO.getQuantity());
        warehouseInventoryItem.setCompanyId(itemDTO.getCompanyId());
    }
}
