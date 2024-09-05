package org.chainoptimstorage.core.warehouseinventoryitem.service;

import org.chainoptimstorage.core.warehouseinventoryitem.dto.CreateWarehouseInventoryItemDTO;
import org.chainoptimstorage.core.warehouseinventoryitem.dto.UpdateWarehouseInventoryItemDTO;
import org.chainoptimstorage.core.warehouseinventoryitem.model.WarehouseInventoryItem;
import org.chainoptimstorage.shared.PaginatedResults;
import org.chainoptimstorage.shared.enums.SearchMode;
import org.chainoptimstorage.shared.search.SearchParams;

import java.util.List;

public interface WarehouseInventoryItemService {

    List<WarehouseInventoryItem> getWarehouseInventoryItemsByOrganizationId(Integer organizationId);
    List<WarehouseInventoryItem> getWarehouseInventoryItemsByWarehouseId(Integer warehouseId);
    PaginatedResults<WarehouseInventoryItem> getWarehouseInventoryItemsAdvanced(SearchMode searchMode, Integer entity, SearchParams searchParams);
    Integer getOrganizationIdById(Long warehouseInventoryItemId);
    long countByOrganizationId(Integer organizationId);

    WarehouseInventoryItem createWarehouseInventoryItem(CreateWarehouseInventoryItemDTO order);
    List<WarehouseInventoryItem> createWarehouseInventoryItemsInBulk(List<CreateWarehouseInventoryItemDTO> orderDTOs);
    List<WarehouseInventoryItem> updateWarehouseInventoryItemsInBulk(List<UpdateWarehouseInventoryItemDTO> orderDTOs);
    List<Integer> deleteWarehouseInventoryItemsInBulk(List<Integer> orders);
}
