package org.chainoptim.internalcommunication.in.supplier.repository;

import org.chainoptim.internalcommunication.in.supplier.model.Supplier;

import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface SupplierRepository {

    List<Supplier> findSuppliersByOrganizationId(Integer organizationId);
    Optional<Integer> findOrganizationIdById(Long supplierId);
    long countByOrganizationId(@Param("organizationId") Integer organizationId);
}
