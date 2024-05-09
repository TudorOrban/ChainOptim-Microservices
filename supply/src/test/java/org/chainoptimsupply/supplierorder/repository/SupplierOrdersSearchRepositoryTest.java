package org.chainoptimsupply.supplierorder.repository;

import org.chainoptimsupply.core.supplierorder.repository.SupplierOrdersSearchRepositoryImpl;
import org.chainoptimsupply.shared.PaginatedResults;
import org.chainoptimsupply.shared.enums.SearchMode;
import org.chainoptimsupply.shared.search.SearchParams;
import org.chainoptimsupply.core.supplier.model.Supplier;
import org.chainoptimsupply.core.supplierorder.model.OrderStatus;
import org.chainoptimsupply.core.supplierorder.model.SupplierOrder;

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
class SupplierOrdersSearchRepositoryTest {

    @Autowired
    private SupplierOrdersSearchRepositoryImpl supplierOrdersSearchRepository;

    @Autowired
    private TestEntityManager entityManager;

    @BeforeEach
    void setUp() {
        createTestSupplier();
        Integer componentId = 1;
        createTestSupplierOrder("no 1", "2024-01-23 12:02:02", "2024-01-23 12:02:02", componentId, OrderStatus.DELIVERED, 1f);
        createTestSupplierOrder("no 12", "2024-02-23 12:02:02", "2024-02-23 12:02:02", componentId, OrderStatus.DELIVERED, 2f);
        createTestSupplierOrder("no 3", "2024-03-23 12:02:02", "2024-03-23 12:02:02", componentId, OrderStatus.DELIVERED, 3f);
        createTestSupplierOrder("no 4", "2024-04-23 12:02:02", "2024-04-23 12:02:02", componentId, OrderStatus.PLACED, 4f);
        createTestSupplierOrder("no 5", "2024-05-23 12:02:02", "2024-05-23 12:02:02", componentId, OrderStatus.PLACED, 5f);
        createTestSupplierOrder("no 6", "2024-06-23 12:02:02", "2024-06-23 12:02:02", componentId, OrderStatus.CANCELED, 6f);
        entityManager.flush();
    }

    @Test
    void findByOrganizationIdAdvanced_SearchQueryWorks() {
        // Arrange
        Integer supplierId = 1;
        String searchQuery = "no 1";
        Map<String, String> filters = new HashMap<>();
        String sortBy = "orderDate";
        boolean ascending = true;
        int page = 1;
        int itemsPerPage = 10;

        // Act
        PaginatedResults<SupplierOrder> paginatedResults = supplierOrdersSearchRepository.findBySupplierIdAdvanced(
                SearchMode.SECONDARY, supplierId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(2, paginatedResults.results.size());

        // Arrange
        searchQuery = "Non-valid";

        // Act
        PaginatedResults<SupplierOrder> newPaginatedResults = supplierOrdersSearchRepository.findBySupplierIdAdvanced(
                SearchMode.SECONDARY, supplierId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(0, newPaginatedResults.results.size());
    }

    @Test
    void findByOrganizationIdAdvanced_PaginationWorks() {
        // Arrange
        Integer supplierId = 1;
        String searchQuery = "";
        Map<String, String> filters = new HashMap<>();
        String sortBy = "orderDate";
        boolean ascending = true;
        int page = 1;
        int itemsPerPage = 4;

        // Act
        PaginatedResults<SupplierOrder> paginatedResults = supplierOrdersSearchRepository.findBySupplierIdAdvanced(
                SearchMode.SECONDARY, supplierId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(4, paginatedResults.results.size());

        // Arrange
        page = 2;

        // Act
        PaginatedResults<SupplierOrder> newPaginatedResults = supplierOrdersSearchRepository.findBySupplierIdAdvanced(
                SearchMode.SECONDARY, supplierId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(2, newPaginatedResults.results.size());

        assertEquals(6, paginatedResults.totalCount);
    }

    @Test
    void findByOrganizationIdAdvanced_SortOptionsWork() {
        // Arrange
        Integer supplierId = 1;
        String searchQuery = "";
        Map<String, String> filters = new HashMap<>();
        String sortBy = "orderDate";
        boolean ascending = true;
        int page = 1;
        int itemsPerPage = 10;

        // Act
        PaginatedResults<SupplierOrder> paginatedResults = supplierOrdersSearchRepository.findBySupplierIdAdvanced(
                SearchMode.SECONDARY, supplierId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(1, paginatedResults.results.getFirst().getId());

        // Arrange
        ascending = false;

        // Act
        PaginatedResults<SupplierOrder> newPaginatedResults = supplierOrdersSearchRepository.findBySupplierIdAdvanced(
                SearchMode.SECONDARY, supplierId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(6, newPaginatedResults.results.getFirst().getId());

        // Arrange
        sortBy = "deliveryDate";
        ascending = true;

        // Act
        PaginatedResults<SupplierOrder> newPaginatedResults2 = supplierOrdersSearchRepository.findBySupplierIdAdvanced(
                SearchMode.SECONDARY, supplierId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(1, newPaginatedResults2.results.getFirst().getId());

        // Arrange
        sortBy = "companyId";

        // Act
        PaginatedResults<SupplierOrder> newPaginatedResults3 = supplierOrdersSearchRepository.findBySupplierIdAdvanced(
                SearchMode.SECONDARY, supplierId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(1, newPaginatedResults3.results.getFirst().getId());
    }
    
    @Test
    void findByOrganizationIdAdvanced_FiltersWork() {
        // Arrange
        Integer supplierId = 1;
        String searchQuery = "";
        Map<String, String> filters = new HashMap<>();
        filters.put("status", "DELIVERED");
        String sortBy = "orderDate";
        boolean ascending = true;
        int page = 1;
        int itemsPerPage = 10;

        // Act
        PaginatedResults<SupplierOrder> paginatedResults = supplierOrdersSearchRepository.findBySupplierIdAdvanced(
                SearchMode.SECONDARY, supplierId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(3, paginatedResults.results.size());

        // Arrange
        filters.clear();
        filters.put("orderDateStart", "2024-03-01T00:00:00");

        // Act
        PaginatedResults<SupplierOrder> newPaginatedResults = supplierOrdersSearchRepository.findBySupplierIdAdvanced(
                SearchMode.SECONDARY, supplierId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(4, newPaginatedResults.results.size());

        // Arrange
        filters.clear();
        filters.put("orderDateEnd", "2024-03-01T00:00:00");

        // Act
        PaginatedResults<SupplierOrder> newPaginatedResults2 = supplierOrdersSearchRepository.findBySupplierIdAdvanced(
                SearchMode.SECONDARY, supplierId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(2, newPaginatedResults2.results.size());

        // Arrange
        filters.clear();
        filters.put("greaterThanQuantity", "3");

        // Act
        PaginatedResults<SupplierOrder> newPaginatedResults3 = supplierOrdersSearchRepository.findBySupplierIdAdvanced(
                SearchMode.SECONDARY, supplierId, new SearchParams(searchQuery, null, filters, sortBy, ascending, page, itemsPerPage)
        );

        // Assert
        assertEquals(4, newPaginatedResults3.results.size());
    }

    private void createTestSupplier() {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

        Supplier supplier = Supplier.builder()
                .organizationId(1)
                .name("Test Supplier")
                .createdAt(LocalDateTime.parse("2024-01-23 12:02:02", formatter))
                .updatedAt(LocalDateTime.parse("2024-01-23 12:02:02", formatter))
                .locationId(1)
                .build();

        entityManager.persist(supplier);
    }

    private void createTestSupplierOrder(String companyId, String orderDate, String deliveryDate, Integer componentId, OrderStatus status, float quantity) {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        SupplierOrder supplierOrder = SupplierOrder.builder()
                .organizationId(1)
                .supplierId(1)
                .companyId(companyId)
                .orderDate(LocalDateTime.parse(orderDate, formatter))
                .deliveryDate(LocalDateTime.parse(deliveryDate, formatter))
                .componentId(componentId)
                .status(status)
                .quantity(quantity)
                .build();

        entityManager.persist(supplierOrder);
    }
}
