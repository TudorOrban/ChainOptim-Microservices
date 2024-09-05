package org.chainoptimstorage.internal.out.warehouse.controller;

import org.chainoptimstorage.core.warehouse.model.Warehouse;
import org.chainoptimstorage.core.warehouse.service.WarehouseService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/v1/internal/warehouses")
public class WarehouseInternalController {

    private final WarehouseService warehouseService;

    @Autowired
    public WarehouseInternalController(WarehouseService warehouseService) {
        this.warehouseService = warehouseService;
    }

    @GetMapping("/organization/{organizationId}")
    public List<Warehouse> getWarehousesByOrganizationId(@PathVariable Integer organizationId) {
        return warehouseService.getWarehousesByOrganizationId(organizationId);
    }

    @GetMapping("/organization/{organizationId}/count")
    public long countByOrganizationId(@PathVariable Integer organizationId) {
        return warehouseService.countByOrganizationId(organizationId);
    }

    @GetMapping("/{warehouseId}/organization-id")
    public Integer getOrganizationIdById(@PathVariable Long warehouseId) {
        return warehouseService.getOrganizationIdById(warehouseId);
    }
}
