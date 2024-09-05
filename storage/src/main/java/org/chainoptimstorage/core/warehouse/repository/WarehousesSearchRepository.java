package org.chainoptimstorage.core.warehouse.repository;


import org.chainoptimstorage.core.warehouse.model.Warehouse;
import org.chainoptimstorage.shared.PaginatedResults;

public interface WarehousesSearchRepository {
    PaginatedResults<Warehouse> findByOrganizationIdAdvanced(
            Integer organizationId,
            String searchQuery,
            String sortBy,
            boolean ascending,
            int page,
            int itemsPerPage
    );
}
