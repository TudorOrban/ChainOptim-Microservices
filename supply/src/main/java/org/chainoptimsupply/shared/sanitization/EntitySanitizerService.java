package org.chainoptimsupply.shared.sanitization;

import org.chainoptimsupply.supplier.dto.*;

public interface EntitySanitizerService {

    // Supplier
    CreateSupplierDTO sanitizeCreateSupplierDTO(CreateSupplierDTO supplierDTO);
    UpdateSupplierDTO sanitizeUpdateSupplierDTO(UpdateSupplierDTO supplierDTO);
    CreateSupplierOrderDTO sanitizeCreateSupplierOrderDTO(CreateSupplierOrderDTO orderDTO);
    UpdateSupplierOrderDTO sanitizeUpdateSupplierOrderDTO(UpdateSupplierOrderDTO orderDTO);
    CreateSupplierShipmentDTO sanitizeCreateSupplierShipmentDTO(CreateSupplierShipmentDTO shipmentDTO);
    UpdateSupplierShipmentDTO sanitizeUpdateSupplierShipmentDTO(UpdateSupplierShipmentDTO shipmentDTO);

}
