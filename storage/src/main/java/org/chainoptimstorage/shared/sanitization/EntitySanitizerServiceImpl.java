package org.chainoptimstorage.shared.sanitization;

import org.chainoptimstorage.core.warehouse.dto.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class EntitySanitizerServiceImpl implements EntitySanitizerService {

    private final SanitizationService sanitizationService;

    @Autowired
    public EntitySanitizerServiceImpl(SanitizationService sanitizationService) {
        this.sanitizationService = sanitizationService;
    }


    // Suppliers
    public CreateWarehouseDTO sanitizeCreateSupplierDTO(CreateWarehouseDTO supplierDTO) {
        supplierDTO.setName(sanitizationService.sanitize(supplierDTO.getName()));

        return supplierDTO;
    }

    public UpdateWarehouseDTO sanitizeUpdateSupplierDTO(UpdateWarehouseDTO supplierDTO) {
        supplierDTO.setName(sanitizationService.sanitize(supplierDTO.getName()));

        return supplierDTO;
    }

    public CreateSupplierOrderDTO sanitizeCreateSupplierOrderDTO(CreateSupplierOrderDTO orderDTO) {
        return orderDTO;
    }

    public UpdateSupplierOrderDTO sanitizeUpdateSupplierOrderDTO(UpdateSupplierOrderDTO orderDTO) {
        return orderDTO;
    }

    public CreateSupplierShipmentDTO sanitizeCreateSupplierShipmentDTO(CreateSupplierShipmentDTO shipmentDTO) {
        return shipmentDTO;
    }

    public UpdateSupplierShipmentDTO sanitizeUpdateSupplierShipmentDTO(UpdateSupplierShipmentDTO shipmentDTO) {
        return shipmentDTO;
    }

}
