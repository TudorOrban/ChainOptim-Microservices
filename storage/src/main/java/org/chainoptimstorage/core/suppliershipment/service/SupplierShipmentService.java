package org.chainoptimstorage.core.suppliershipment.service;


import org.chainoptimstorage.core.warehouse.dto.CreateSupplierShipmentDTO;
import org.chainoptimstorage.core.warehouse.dto.UpdateSupplierShipmentDTO;
import org.chainoptimstorage.shared.PaginatedResults;
import org.chainoptimstorage.core.suppliershipment.model.SupplierShipment;

import java.util.List;

public interface SupplierShipmentService {

    List<SupplierShipment> getSupplierShipmentsBySupplierOrderId(Integer orderId);
    List<SupplierShipment> getSupplierShipmentsBySupplierOrderIds(List<Integer> orderIds);
    PaginatedResults<SupplierShipment> getSupplierShipmentsBySupplierOrderIdAdvanced(Integer supplierOrderId, String searchQuery, String sortBy, boolean ascending, int page, int itemsPerPage);
    SupplierShipment getSupplierShipmentById(Integer shipmentId);
    long countByOrganizationId(Integer organizationId);
    SupplierShipment createSupplierShipment(CreateSupplierShipmentDTO shipmentDTO);
    List<SupplierShipment> createSupplierShipmentsInBulk(List<CreateSupplierShipmentDTO> shipmentDTOs);
    List<SupplierShipment> updateSupplierShipmentsInBulk(List<UpdateSupplierShipmentDTO> shipmentDTOs);
    List<Integer> deleteSupplierShipmentsInBulk(List<Integer> shipmentIds);
}
