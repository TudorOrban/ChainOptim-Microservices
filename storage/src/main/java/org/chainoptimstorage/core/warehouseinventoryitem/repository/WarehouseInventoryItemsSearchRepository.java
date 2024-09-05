package org.chainoptimstorage.core.warehouseinventoryitem.repository;

import org.chainoptimstorage.core.warehouseinventoryitem.model.WarehouseInventoryItem;
import org.chainoptimstorage.shared.PaginatedResults;
import org.chainoptimstorage.shared.enums.SearchMode;
import org.chainoptimstorage.shared.search.SearchParams;

public interface WarehouseInventoryItemsSearchRepository {
    PaginatedResults<WarehouseInventoryItem> findByWarehouseIdAdvanced(
            SearchMode searchMode,
            Integer warehouseId,
            SearchParams searchParams
    );
}
