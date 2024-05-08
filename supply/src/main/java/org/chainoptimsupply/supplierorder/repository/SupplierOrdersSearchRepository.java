package org.chainoptimsupply.supplierorder.repository;

import org.chainoptimsupply.shared.PaginatedResults;
import org.chainoptimsupply.shared.enums.SearchMode;
import org.chainoptimsupply.shared.search.SearchParams;
import org.chainoptimsupply.supplierorder.model.SupplierOrder;

public interface SupplierOrdersSearchRepository {
    PaginatedResults<SupplierOrder> findBySupplierIdAdvanced(
            SearchMode searchMode,
            Integer supplierId,
            SearchParams searchParams
    );
}
