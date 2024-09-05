package org.chainoptimstorage.core.suppliershipment.repository;


import org.chainoptimstorage.shared.PaginatedResults;
import org.chainoptimstorage.core.suppliershipment.model.SupplierShipment;

public interface SupplierShipmentsSearchRepository {
    PaginatedResults<SupplierShipment> findBySupplierOrderIdAdvanced(
            Integer supplierOrderId,
            String searchQuery,
            String sortBy,
            boolean ascending,
            int page,
            int itemsPerPage
    );
}
