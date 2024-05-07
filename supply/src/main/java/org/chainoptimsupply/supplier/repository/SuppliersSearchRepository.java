package org.chainoptimsupply.supplier.repository;


import org.chainoptimsupply.shared.PaginatedResults;
import org.chainoptimsupply.supplier.model.Supplier;

public interface SuppliersSearchRepository {
    PaginatedResults<Supplier> findByOrganizationIdAdvanced(
            Integer organizationId,
            String searchQuery,
            String sortBy,
            boolean ascending,
            int page,
            int itemsPerPage
    );
}
