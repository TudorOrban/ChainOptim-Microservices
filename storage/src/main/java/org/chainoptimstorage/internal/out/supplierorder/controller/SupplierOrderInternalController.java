package org.chainoptimstorage.internal.out.supplierorder.controller;

import org.chainoptimstorage.core.supplierorder.model.SupplierOrder;
import org.chainoptimstorage.core.supplierorder.service.SupplierOrderService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/v1/internal/supplier-order")
public class SupplierOrderInternalController {

    private final SupplierOrderService supplierOrderService;

    @Autowired
    public SupplierOrderInternalController(SupplierOrderService supplierOrderService) {
        this.supplierOrderService = supplierOrderService;
    }

    @GetMapping("/organization/{organizationId}")
    public List<SupplierOrder> getSupplierOrdersByOrganizationId(@PathVariable Integer organizationId) {
        return supplierOrderService.getSupplierOrdersByOrganizationId(organizationId);
    }

    @GetMapping("/organization/{organizationId}/count")
    public long countByOrganizationId(@PathVariable Integer organizationId) {
        return supplierOrderService.countByOrganizationId(organizationId);
    }

    @GetMapping("/{supplierOrderId}/organization-id")
    public Integer getOrganizationIdById(@PathVariable Long supplierOrderId) {
        return supplierOrderService.getOrganizationIdById(supplierOrderId);
    }
}
