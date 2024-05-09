package org.chainoptim.features.supplier.repository;

import org.chainoptim.features.supplier.model.SupplierOrder;

import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface SupplierOrderRepositoryNew {

    List<SupplierOrder> findSupplierOrdersByOrganizationId(Integer organizationId);
    Optional<Integer> findOrganizationIdById(Long supplierOrderId);
    long countByOrganizationId(@Param("organizationId") Integer organizationId);
}
