package org.chainoptimsupply.suppliershipment.repository;


import org.chainoptimsupply.shared.PaginatedResults;
import org.chainoptimsupply.suppliershipment.model.SupplierShipment;

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
