package org.chainoptimsupply.core.supplier.repository;


import org.chainoptimsupply.core.supplier.model.Supplier;
import org.chainoptimsupply.shared.PaginatedResults;

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
