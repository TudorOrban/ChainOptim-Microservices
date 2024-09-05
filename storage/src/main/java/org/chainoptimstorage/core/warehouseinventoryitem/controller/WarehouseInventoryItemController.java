package org.chainoptimstorage.core.warehouseinventoryitem.controller;

import org.chainoptimstorage.core.warehouseinventoryitem.dto.CreateWarehouseInventoryItemDTO;
import org.chainoptimstorage.core.warehouseinventoryitem.dto.UpdateWarehouseInventoryItemDTO;
import org.chainoptimstorage.internal.in.security.service.SecurityService;
import org.chainoptimstorage.core.warehouseinventoryitem.model.WarehouseInventoryItem;
import org.chainoptimstorage.core.warehouseinventoryitem.service.WarehouseInventoryItemService;
import org.chainoptimstorage.shared.PaginatedResults;
import org.chainoptimstorage.shared.enums.SearchMode;
import org.chainoptimstorage.shared.search.SearchParams;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/v1/warehouse-inventory-items")
public class WarehouseInventoryItemController {

    private final WarehouseInventoryItemService warehouseInventoryItemService;
    private final SecurityService securityService;

    @Autowired
    public WarehouseInventoryItemController(WarehouseInventoryItemService warehouseInventoryItemService,
                                            SecurityService securityService) {
        this.warehouseInventoryItemService = warehouseInventoryItemService;
        this.securityService = securityService;
    }

    @PreAuthorize("@securityService.canAccessOrganizationEntity(#organizationId, \"Warehouse\", \"Read\")")
    @GetMapping("/organization/{organizationId}")
    public ResponseEntity<List<WarehouseInventoryItem>> getWarehouseInventoryItemsByOrganizationId(@PathVariable Integer organizationId) {
        List<WarehouseInventoryItem> warehouseInventoryItems = warehouseInventoryItemService.getWarehouseInventoryItemsByOrganizationId(organizationId);
        return ResponseEntity.ok(warehouseInventoryItems);
    }

//    @PreAuthorize("@securityService.canAccessOrganizationEntity(#organizationId, \"Organization\", \"Read\")")
    @GetMapping("/organization/advanced/{organizationId}")
    public ResponseEntity<PaginatedResults<WarehouseInventoryItem>> getWarehouseInventoryItemsByOrganizationIdAdvanced(
            @PathVariable Integer organizationId,
            @RequestParam(name = "searchQuery", required = false, defaultValue = "") String searchQuery,
            @RequestParam(name = "filters", required = false, defaultValue = "") String filtersJson,
            @RequestParam(name = "sortBy", required = false, defaultValue = "createdAt") String sortBy,
            @RequestParam(name = "ascending", required = false, defaultValue = "true") boolean ascending,
            @RequestParam(name = "page", required = false, defaultValue = "1") int page,
            @RequestParam(name = "itemsPerPage", required = false, defaultValue = "30") int itemsPerPage
    ) {
        SearchParams searchParams = new SearchParams(searchQuery, filtersJson, null, sortBy, ascending, page, itemsPerPage);
        PaginatedResults<WarehouseInventoryItem> warehouseInventoryItems = warehouseInventoryItemService.getWarehouseInventoryItemsAdvanced(SearchMode.ORGANIZATION, organizationId, searchParams);
        return ResponseEntity.ok(warehouseInventoryItems);
    }

    @PreAuthorize("@securityService.canAccessEntity(#warehouseId, \"Warehouse\", \"Read\")")
    @GetMapping("/warehouse/advanced/{warehouseId}")
    public ResponseEntity<PaginatedResults<WarehouseInventoryItem>> getWarehouseInventoryItemsByWarehouseIdAdvanced(
            @PathVariable Integer warehouseId,
            @RequestParam(name = "searchQuery", required = false, defaultValue = "") String searchQuery,
            @RequestParam(name = "filters", required = false, defaultValue = "") String filtersJson,
            @RequestParam(name = "sortBy", required = false, defaultValue = "createdAt") String sortBy,
            @RequestParam(name = "ascending", required = false, defaultValue = "true") boolean ascending,
            @RequestParam(name = "page", required = false, defaultValue = "1") int page,
            @RequestParam(name = "itemsPerPage", required = false, defaultValue = "30") int itemsPerPage
    ) {
        SearchParams searchParams = new SearchParams(searchQuery, filtersJson, null, sortBy, ascending, page, itemsPerPage);
        PaginatedResults<WarehouseInventoryItem> warehouseInventoryItems = warehouseInventoryItemService.getWarehouseInventoryItemsAdvanced(SearchMode.SECONDARY, warehouseId, searchParams);
        return ResponseEntity.ok(warehouseInventoryItems);
    }

    // Create
    @PreAuthorize("@securityService.canAccessOrganizationEntity(#itemDTO.getOrganizationId(), \"WarehouseInventoryItem\", \"Create\")")
    @PostMapping("/create")
    public ResponseEntity<WarehouseInventoryItem> createWarehouseInventoryItem(@RequestBody CreateWarehouseInventoryItemDTO itemDTO) {
        WarehouseInventoryItem warehouseInventoryItem = warehouseInventoryItemService.createWarehouseInventoryItem(itemDTO);
        return ResponseEntity.ok(warehouseInventoryItem);
    }

    @PreAuthorize("@securityService.canAccessOrganizationEntity(#itemDTOs.getFirst().getOrganizationId(), \"WarehouseInventoryItem\", \"Create\")")
    @PostMapping("/create/bulk")
    public ResponseEntity<List<WarehouseInventoryItem>> createWarehouseInventoryItemsInBulk(@RequestBody List<CreateWarehouseInventoryItemDTO> itemDTOs) {
        List<WarehouseInventoryItem> warehouseInventoryItems = warehouseInventoryItemService.createWarehouseInventoryItemsInBulk(itemDTOs);
        return ResponseEntity.ok(warehouseInventoryItems);
    }

    // Update
    @PreAuthorize("@securityService.canAccessOrganizationEntity(#itemDTOs.getFirst().getOrganizationId(), \"WarehouseInventoryItem\", \"Update\")")
    @PutMapping("/update/bulk")
    public ResponseEntity<List<WarehouseInventoryItem>> updateWarehouseInventoryItemsInBulk(@RequestBody List<UpdateWarehouseInventoryItemDTO> itemDTOs) {
        List<WarehouseInventoryItem> warehouseInventoryItems = warehouseInventoryItemService.updateWarehouseInventoryItemsInBulk(itemDTOs);
        return ResponseEntity.ok(warehouseInventoryItems);
    }

    // Delete
    @PreAuthorize("@securityService.canAccessEntity(#itemIds.getFirst(), \"WarehouseInventoryItem\", \"Delete\")") // Secure as service method ensures all items belong to the same organization
    @DeleteMapping("/delete/bulk")
    public ResponseEntity<List<Integer>> deleteWarehouseInventoryItemsInBulk(@RequestBody List<Integer> itemIds) {
        warehouseInventoryItemService.deleteWarehouseInventoryItemsInBulk(itemIds);

        return ResponseEntity.ok(itemIds);
    }

}