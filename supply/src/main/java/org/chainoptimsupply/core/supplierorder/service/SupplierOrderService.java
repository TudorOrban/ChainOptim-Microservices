package org.chainoptimsupply.core.supplierorder.service;

import org.chainoptimsupply.core.supplierorder.model.SupplierOrder;
import org.chainoptimsupply.shared.PaginatedResults;
import org.chainoptimsupply.shared.enums.SearchMode;
import org.chainoptimsupply.shared.search.SearchParams;
import org.chainoptimsupply.core.supplier.dto.CreateSupplierOrderDTO;
import org.chainoptimsupply.core.supplier.dto.UpdateSupplierOrderDTO;

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
