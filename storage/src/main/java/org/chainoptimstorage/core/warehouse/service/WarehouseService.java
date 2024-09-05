package org.chainoptimstorage.core.warehouse.service;


import org.chainoptimstorage.core.warehouse.dto.CreateWarehouseDTO;
import org.chainoptimstorage.core.warehouse.dto.WarehousesSearchDTO;
import org.chainoptimstorage.core.warehouse.dto.UpdateWarehouseDTO;
import org.chainoptimstorage.core.warehouse.model.Warehouse;
import org.chainoptimstorage.shared.PaginatedResults;

import java.util.List;

public interface WarehouseService {
    // Fetch
    List<Warehouse> getAllWarehouses();
    Warehouse getWarehouseById(Integer id);
    List<Warehouse> getWarehousesByOrganizationId(Integer organizationId);
    PaginatedResults<WarehousesSearchDTO> getWarehousesByOrganizationIdAdvanced(Integer organizationId, String searchQuery, String sortBy, boolean ascending, int page, int itemsPerPage);
    Integer getOrganizationIdById(Long warehouseId);
    long countByOrganizationId(Integer organizationId);

    // Create
    Warehouse createWarehouse(CreateWarehouseDTO warehouseDTO);

    // Update
    Warehouse updateWarehouse(UpdateWarehouseDTO updateWarehouseDTO);

    // Delete
    void deleteWarehouse(Integer warehouseId);
}
