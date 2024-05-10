package org.chainoptim.internalcommunication.in.supplier.repository;

import org.chainoptim.internalcommunication.in.supplier.model.SupplierOrder;

import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface SupplierOrderRepository {

    List<SupplierOrder> findSupplierOrdersByOrganizationId(Integer organizationId);
    Optional<Integer> findOrganizationIdById(Long supplierOrderId);
    long countByOrganizationId(@Param("organizationId") Integer organizationId);
}
