package org.chainoptimsupply.core.supplierorder.repository;

import org.chainoptimsupply.core.supplierorder.model.SupplierOrder;
import org.chainoptimsupply.shared.PaginatedResults;
import org.chainoptimsupply.shared.enums.SearchMode;
import org.chainoptimsupply.shared.search.SearchParams;

public interface SupplierOrdersSearchRepository {
    PaginatedResults<SupplierOrder> findBySupplierIdAdvanced(
            SearchMode searchMode,
            Integer supplierId,
            SearchParams searchParams
    );
}
