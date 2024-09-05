package org.chainoptim.internalcommunication.in.storage.repository;

import org.chainoptim.internalcommunication.in.storage.model.Warehouse;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface WarehouseRepository {

    List<Warehouse> findWarehousesByOrganizationId(Integer organizationId);
    Optional<Integer> findOrganizationIdById(Long warehouseId);
    long countByOrganizationId(@Param("organizationId") Integer organizationId);
}
