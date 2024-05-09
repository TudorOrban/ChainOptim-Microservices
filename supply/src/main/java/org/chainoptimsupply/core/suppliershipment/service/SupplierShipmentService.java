package org.chainoptimsupply.core.suppliershipment.service;


import org.chainoptimsupply.core.supplier.dto.CreateSupplierShipmentDTO;
import org.chainoptimsupply.core.supplier.dto.UpdateSupplierShipmentDTO;
import org.chainoptimsupply.shared.PaginatedResults;
import org.chainoptimsupply.core.suppliershipment.model.SupplierShipment;

import java.util.List;

public interface SupplierShipmentService {

    List<SupplierShipment> getSupplierShipmentBySupplierOrderId(Integer orderId);
    PaginatedResults<SupplierShipment> getSupplierShipmentsBySupplierOrderIdAdvanced(Integer supplierOrderId, String searchQuery, String sortBy, boolean ascending, int page, int itemsPerPage);
    SupplierShipment getSupplierShipmentById(Integer shipmentId);
    SupplierShipment createSupplierShipment(CreateSupplierShipmentDTO shipmentDTO);
    List<SupplierShipment> createSupplierShipmentsInBulk(List<CreateSupplierShipmentDTO> shipmentDTOs);
    List<SupplierShipment> updateSupplierShipmentsInBulk(List<UpdateSupplierShipmentDTO> shipmentDTOs);
    List<Integer> deleteSupplierShipmentsInBulk(List<Integer> shipmentIds);
}
