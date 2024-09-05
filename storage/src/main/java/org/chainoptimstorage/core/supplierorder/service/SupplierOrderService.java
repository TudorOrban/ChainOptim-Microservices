package org.chainoptimstorage.core.supplierorder.service;

import org.chainoptimstorage.core.supplierorder.model.SupplierOrder;
import org.chainoptimstorage.shared.PaginatedResults;
import org.chainoptimstorage.shared.enums.SearchMode;
import org.chainoptimstorage.shared.search.SearchParams;
import org.chainoptimstorage.core.warehouse.dto.CreateSupplierOrderDTO;
import org.chainoptimstorage.core.warehouse.dto.UpdateSupplierOrderDTO;

import java.util.List;

public interface SupplierOrderService {

    List<SupplierOrder> getSupplierOrdersByOrganizationId(Integer organizationId);
    List<SupplierOrder> getSupplierOrdersBySupplierId(Integer supplierId);
    PaginatedResults<SupplierOrder> getSupplierOrdersAdvanced(SearchMode searchMode, Integer entity, SearchParams searchParams);
    Integer getOrganizationIdById(Long supplierOrderId);
    long countByOrganizationId(Integer organizationId);

    SupplierOrder createSupplierOrder(CreateSupplierOrderDTO order);
    List<SupplierOrder> createSupplierOrdersInBulk(List<CreateSupplierOrderDTO> orderDTOs);
    List<SupplierOrder> updateSuppliersOrdersInBulk(List<UpdateSupplierOrderDTO> orderDTOs);
    List<Integer> deleteSupplierOrdersInBulk(List<Integer> orders);
}
