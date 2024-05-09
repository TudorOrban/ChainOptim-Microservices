package org.chainoptimsupply.core.suppliershipment.repository;


import org.chainoptimsupply.shared.PaginatedResults;
import org.chainoptimsupply.core.suppliershipment.model.SupplierShipment;

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
