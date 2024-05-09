package org.chainoptimsupply.internal.out.suppliershipment.controller;

import org.chainoptimsupply.core.suppliershipment.model.SupplierShipment;
import org.chainoptimsupply.core.suppliershipment.service.SupplierShipmentService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/v1/internal/supplier-shipments")
public class SupplierShipmentInternalController {

    private final SupplierShipmentService supplierShipmentService;

    @Autowired
    public SupplierShipmentInternalController(SupplierShipmentService supplierShipmentService) {
        this.supplierShipmentService = supplierShipmentService;
    }

    @GetMapping("/supplier-orders/{orderIds}")
    public List<SupplierShipment> getSupplierShipmentsBySupplierOrderIds(@PathVariable List<Integer> orderIds) {
        return supplierShipmentService.getSupplierShipmentsBySupplierOrderIds(orderIds);
    }

    @GetMapping("/organization/{organizationId}/count")
    public long getSupplierShipmentsByOrganizationId(@PathVariable Integer organizationId) {
        return supplierShipmentService.countByOrganizationId(organizationId);
    }
}
