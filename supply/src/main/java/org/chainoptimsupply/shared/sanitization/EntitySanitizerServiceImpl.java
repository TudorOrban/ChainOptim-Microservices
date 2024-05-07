package org.chainoptimsupply.shared.sanitization;

import org.chainoptimsupply.supplier.dto.*;
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
    public CreateSupplierDTO sanitizeCreateSupplierDTO(CreateSupplierDTO supplierDTO) {
        supplierDTO.setName(sanitizationService.sanitize(supplierDTO.getName()));

        return supplierDTO;
    }

    public UpdateSupplierDTO sanitizeUpdateSupplierDTO(UpdateSupplierDTO supplierDTO) {
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
