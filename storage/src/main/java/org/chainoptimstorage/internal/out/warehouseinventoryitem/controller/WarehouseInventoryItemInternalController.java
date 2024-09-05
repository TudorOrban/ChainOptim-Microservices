package org.chainoptimstorage.internal.out.warehouseinventoryitem.controller;

import org.chainoptimstorage.core.warehouseinventoryitem.model.WarehouseInventoryItem;
import org.chainoptimstorage.core.warehouseinventoryitem.service.WarehouseInventoryItemService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/v1/internal/warehouse-inventory-items")
public class WarehouseInventoryItemInternalController {

    private final WarehouseInventoryItemService warehouseInventoryItemService;

    @Autowired
    public WarehouseInventoryItemInternalController(WarehouseInventoryItemService warehouseInventoryItemService) {
        this.warehouseInventoryItemService = warehouseInventoryItemService;
    }

    @GetMapping("/organization/{organizationId}")
    public List<WarehouseInventoryItem> getWarehouseInventoryItemsByOrganizationId(@PathVariable Integer organizationId) {
        return warehouseInventoryItemService.getWarehouseInventoryItemsByOrganizationId(organizationId);
    }

    @GetMapping("/organization/{organizationId}/count")
    public long countByOrganizationId(@PathVariable Integer organizationId) {
        return warehouseInventoryItemService.countByOrganizationId(organizationId);
    }

    @GetMapping("/{warehouseInventoryItemId}/organization-id")
    public Integer getOrganizationIdById(@PathVariable Long warehouseInventoryItemId) {
        return warehouseInventoryItemService.getOrganizationIdById(warehouseInventoryItemId);
    }
}
