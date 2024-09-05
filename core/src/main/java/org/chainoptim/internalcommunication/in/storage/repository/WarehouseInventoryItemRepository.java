package org.chainoptim.internalcommunication.in.storage.repository;

import org.chainoptim.internalcommunication.in.storage.model.WarehouseInventoryItem;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface WarehouseInventoryItemRepository {

    List<WarehouseInventoryItem> findWarehouseInventoryItemsByOrganizationId(Integer organizationId);
    Optional<Integer> findOrganizationIdById(Long warehouseInventoryItemId);
    long countByOrganizationId(@Param("organizationId") Integer organizationId);
}
