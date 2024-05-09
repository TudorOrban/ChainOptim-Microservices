package org.chainoptimsupply.internal.supplier.controller;

import org.chainoptimsupply.core.supplier.model.Supplier;
import org.chainoptimsupply.core.supplier.service.SupplierService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/v1/internal/suppliers")
public class SupplierInternalController {

    private final SupplierService supplierService;

    @Autowired
    public SupplierInternalController(SupplierService supplierService) {
        this.supplierService = supplierService;
    }

    @GetMapping("/organization/{organizationId}")
    public List<Supplier> getSuppliersByOrganizationId(@PathVariable Integer organizationId) {
        return supplierService.getSuppliersByOrganizationId(organizationId);
    }

    @GetMapping("/organization/{organizationId}/count")
    public long countByOrganizationId(@PathVariable Integer organizationId) {
        return supplierService.countByOrganizationId(organizationId);
    }

    @GetMapping("/{supplierId}/organization-id")
    public Integer getOrganizationIdById(@PathVariable Long supplierId) {
        return supplierService.getOrganizationIdById(supplierId);
    }
}
