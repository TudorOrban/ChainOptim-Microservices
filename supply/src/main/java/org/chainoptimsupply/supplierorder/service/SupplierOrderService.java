package org.chainoptimsupply.supplierorder.service;

import org.chainoptimsupply.shared.PaginatedResults;
import org.chainoptimsupply.shared.enums.SearchMode;
import org.chainoptimsupply.shared.search.SearchParams;
import org.chainoptimsupply.supplier.dto.CreateSupplierOrderDTO;
import org.chainoptimsupply.supplier.dto.UpdateSupplierOrderDTO;
import org.chainoptimsupply.supplierorder.model.SupplierOrder;

import java.util.List;

public interface SupplierOrderService {

    List<SupplierOrder> getSupplierOrdersByOrganizationId(Integer organizationId);
    List<SupplierOrder> getSupplierOrdersBySupplierId(Integer supplierId);
    PaginatedResults<SupplierOrder> getSupplierOrdersAdvanced(SearchMode searchMode, Integer entity, SearchParams searchParams);

    SupplierOrder createSupplierOrder(CreateSupplierOrderDTO order);
    List<SupplierOrder> createSupplierOrdersInBulk(List<CreateSupplierOrderDTO> orderDTOs);
    List<SupplierOrder> updateSuppliersOrdersInBulk(List<UpdateSupplierOrderDTO> orderDTOs);
    List<Integer> deleteSupplierOrdersInBulk(List<Integer> orders);
}
