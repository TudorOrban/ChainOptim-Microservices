package org.chainoptimstorage.warehouseinventoryitem.repository;

import org.chainoptimstorage.core.warehouseinventoryitem.repository.WarehouseInventoryItemsSearchRepositoryImpl;
import org.chainoptimstorage.internal.in.location.dto.Location;
import org.chainoptimstorage.shared.PaginatedResults;
import org.chainoptimstorage.shared.enums.SearchMode;
import org.chainoptimstorage.shared.search.SearchParams;
import org.chainoptimstorage.core.warehouse.model.Warehouse;
import org.chainoptimstorage.core.warehouseinventoryitem.model.OrderStatus;
import org.chainoptimstorage.core.warehouseinventoryitem.model.WarehouseInventoryItem;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.boot.test.autoconfigure.orm.jpa.TestEntityManager;
import org.springframework.test.context.junit.jupiter.SpringExtension;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase.Replace.NONE;

@ExtendWith(SpringExtension.class)
@DataJpaTest
@AutoConfigureTestDatabase(replace = NONE)
class WarehouseOrdersSearchRepositoryTest {

    @Autowired
    private WarehouseInventoryItemsSearchRepositoryImpl warehouseInventoryItemsSearchRepository;

    @Autowired
    private TestEntityManager entityManager;

    @BeforeEach
    void setUp() {
        createTestWarehouse();
        Integer componentId = 1;
        createTestWarehouseInventoryItem("no 1", "2024-01-23 12:02:02", 1f);
        createTestWarehouseInventoryItem("no 12", "2024-02-23 12:02:02", 2f);
        createTestWarehouseInventoryItem("no 3", "2024-03-23 12:02:02", 3f);
        createTestWarehouseInventoryItem("no 4", "2024-04-23 12:02:02", 4f);
        createTestWarehouseInventoryItem("no 5", "2024-05-23 12:02:02", 5f);
        createTestWarehouseInventoryItem("no 6", "2024-06-23 12:02:02", 6f);
        entityManager.flush();
    }

    @Test
    void findByOrganizationIdAdvanced_SearchQueryWorks() {
        // Arrange
        Integer warehouseId = 1;
        String searchQuery = "no 1";
        Map<String, String> filters = new HashMap<>();
        String sortBy = "orderDate";
        boolean ascending = true;
        int page = 1;
        int itemsPerPage = 10;

        // Act
        PaginatedResults<WarehouseInventoryItem> paginatedResults = warehouseInventoryItemsSearchRepository.findByWarehouseIdAdvanced(
                SearchMode.SECONDARY, warehouseId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(2, paginatedResults.results.size());

        // Arrange
        searchQuery = "Non-valid";

        // Act
        PaginatedResults<WarehouseInventoryItem> newPaginatedResults = warehouseInventoryItemsSearchRepository.findByWarehouseIdAdvanced(
                SearchMode.SECONDARY, warehouseId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(0, newPaginatedResults.results.size());
    }

    @Test
    void findByOrganizationIdAdvanced_PaginationWorks() {
        // Arrange
        Integer warehouseId = 1;
        String searchQuery = "";
        Map<String, String> filters = new HashMap<>();
        String sortBy = "orderDate";
        boolean ascending = true;
        int page = 1;
        int itemsPerPage = 4;

        // Act
        PaginatedResults<WarehouseInventoryItem> paginatedResults = warehouseInventoryItemsSearchRepository.findByWarehouseIdAdvanced(
                SearchMode.SECONDARY, warehouseId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(4, paginatedResults.results.size());

        // Arrange
        page = 2;

        // Act
        PaginatedResults<WarehouseInventoryItem> newPaginatedResults = warehouseInventoryItemsSearchRepository.findByWarehouseIdAdvanced(
                SearchMode.SECONDARY, warehouseId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(2, newPaginatedResults.results.size());

        assertEquals(6, paginatedResults.totalCount);
    }

    @Test
    void findByOrganizationIdAdvanced_SortOptionsWork() {
        // Arrange
        Integer warehouseId = 1;
        String searchQuery = "";
        Map<String, String> filters = new HashMap<>();
        String sortBy = "orderDate";
        boolean ascending = true;
        int page = 1;
        int itemsPerPage = 10;

        // Act
        PaginatedResults<WarehouseInventoryItem> paginatedResults = warehouseInventoryItemsSearchRepository.findByWarehouseIdAdvanced(
                SearchMode.SECONDARY, warehouseId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(1, paginatedResults.results.getFirst().getId());

        // Arrange
        ascending = false;

        // Act
        PaginatedResults<WarehouseInventoryItem> newPaginatedResults = warehouseInventoryItemsSearchRepository.findByWarehouseIdAdvanced(
                SearchMode.SECONDARY, warehouseId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(6, newPaginatedResults.results.getFirst().getId());

        // Arrange
        sortBy = "deliveryDate";
        ascending = true;

        // Act
        PaginatedResults<WarehouseInventoryItem> newPaginatedResults2 = warehouseInventoryItemsSearchRepository.findByWarehouseIdAdvanced(
                SearchMode.SECONDARY, warehouseId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(1, newPaginatedResults2.results.getFirst().getId());

        // Arrange
        sortBy = "companyId";

        // Act
        PaginatedResults<WarehouseInventoryItem> newPaginatedResults3 = warehouseInventoryItemsSearchRepository.findByWarehouseIdAdvanced(
                SearchMode.SECONDARY, warehouseId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(1, newPaginatedResults3.results.getFirst().getId());
    }
    
    @Test
    void findByOrganizationIdAdvanced_FiltersWork() {
        // Arrange
        Integer warehouseId = 1;
        String searchQuery = "";
        Map<String, String> filters = new HashMap<>();
        filters.put("status", "DELIVERED");
        String sortBy = "orderDate";
        boolean ascending = true;
        int page = 1;
        int itemsPerPage = 10;

        // Act
        PaginatedResults<WarehouseInventoryItem> paginatedResults = warehouseInventoryItemsSearchRepository.findByWarehouseIdAdvanced(
                SearchMode.SECONDARY, warehouseId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(3, paginatedResults.results.size());

        // Arrange
        filters.clear();
        filters.put("orderDateStart", "2024-03-01T00:00:00");

        // Act
        PaginatedResults<WarehouseInventoryItem> newPaginatedResults = warehouseInventoryItemsSearchRepository.findByWarehouseIdAdvanced(
                SearchMode.SECONDARY, warehouseId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(4, newPaginatedResults.results.size());

        // Arrange
        filters.clear();
        filters.put("orderDateEnd", "2024-03-01T00:00:00");

        // Act
        PaginatedResults<WarehouseInventoryItem> newPaginatedResults2 = warehouseInventoryItemsSearchRepository.findByWarehouseIdAdvanced(
                SearchMode.SECONDARY, warehouseId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(2, newPaginatedResults2.results.size());

        // Arrange
        filters.clear();
        filters.put("greaterThanQuantity", "3");

        // Act
        PaginatedResults<WarehouseInventoryItem> newPaginatedResults3 = warehouseInventoryItemsSearchRepository.findByWarehouseIdAdvanced(
                SearchMode.SECONDARY, warehouseId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(4, newPaginatedResults3.results.size());
    }

    private void createTestWarehouse() {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        Location location = new Location();
        location.setId(1);

        Warehouse warehouse = Warehouse.builder()
                .organizationId(1)
                .name("Test Warehouse")
                .createdAt(LocalDateTime.parse("2024-01-23 12:02:02", formatter))
                .updatedAt(LocalDateTime.parse("2024-01-23 12:02:02", formatter))
                .locationId(location.getId())
                .build();

        entityManager.persist(warehouse);
    }

    private void createTestWarehouseInventoryItem(String companyId, String updatedAt, float quantity) {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        WarehouseInventoryItem warehouseInventoryItem = WarehouseInventoryItem.builder()
                .organizationId(1)
                .warehouseId(1)
                .companyId(companyId)
                .updatedAt(LocalDateTime.parse(updatedAt, formatter))
                .quantity(quantity)
                .build();

        entityManager.persist(warehouseInventoryItem);
    }
}
