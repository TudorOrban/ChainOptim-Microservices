package org.chainoptimstorage.warehouseinventoryitem.repository;

import org.chainoptimstorage.core.warehouseinventoryitem.repository.WarehouseInventoryItemRepository;
import org.chainoptimstorage.core.warehouse.model.Warehouse;
import org.chainoptimstorage.internal.in.location.dto.Location;
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
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase.Replace.NONE;

@ExtendWith(SpringExtension.class)
@DataJpaTest
@AutoConfigureTestDatabase(replace = NONE)
class WarehouseOrdersRepositoryTest {

    @Autowired
    private TestEntityManager entityManager;

    @Autowired
    private WarehouseInventoryItemRepository warehouseInventoryItemRepository;

    Integer organizationId = 1;
    Integer locationId = 1;
    Integer warehouseId;
    Integer warehouseInventoryItemId;
    Integer componentId = 1;
    @BeforeEach
    void setUp() {
        // Set up warehouse for update and delete tests
        Warehouse warehouse = addTestWarehouse();
        warehouseId = warehouse.getId();

        WarehouseInventoryItem warehouseInventoryItem = addTestWarehouseInventoryItem();
        warehouseInventoryItemId = warehouseInventoryItem.getId();
    }

    @Test
    void testCreateWarehouseInventoryItem() {
        // Arrange
        WarehouseInventoryItem warehouseInventoryItem = addTestWarehouseInventoryItem();

        // Act
        WarehouseInventoryItem savedWarehouseInventoryItem = entityManager.persist(warehouseInventoryItem);
        entityManager.flush();

        // Assert
        Optional<WarehouseInventoryItem> foundWarehouseInventoryItemOpt = warehouseInventoryItemRepository.findById(savedWarehouseInventoryItem.getId());
        assertTrue(foundWarehouseInventoryItemOpt.isPresent(), "Warehouse should be found in the database");

        WarehouseInventoryItem foundWarehouseInventoryItem = foundWarehouseInventoryItemOpt.get();
        assertEquals(savedWarehouseInventoryItem.getOrganizationId(), foundWarehouseInventoryItem.getOrganizationId());
        assertEquals(savedWarehouseInventoryItem.getWarehouseId(), foundWarehouseInventoryItem.getWarehouseId());
        assertEquals(savedWarehouseInventoryItem.getUpdatedAt(), foundWarehouseInventoryItem.getUpdatedAt());
        assertEquals(savedWarehouseInventoryItem.getQuantity(), foundWarehouseInventoryItem.getQuantity());
    }

    @Test
    void testUpdateWarehouse() {
        // Arrange
        Optional<WarehouseInventoryItem> warehouseInventoryItemOptional = warehouseInventoryItemRepository.findById(warehouseInventoryItemId); // Id from setUp
        if (warehouseInventoryItemOptional.isEmpty()) {
            fail("Expected an existing warehouse order with id " + warehouseInventoryItemId);
        }

        WarehouseInventoryItem warehouseInventoryItem = warehouseInventoryItemOptional.get();
        warehouseInventoryItem.setQuantity(20f);

        // Act
        WarehouseInventoryItem updatedWarehouseInventoryItem = warehouseInventoryItemRepository.save(warehouseInventoryItem);

        // Assert
        assertNotNull(updatedWarehouseInventoryItem);
        assertEquals(warehouseInventoryItem.getOrganizationId(), updatedWarehouseInventoryItem.getOrganizationId());
        assertEquals(warehouseInventoryItem.getWarehouseId(), updatedWarehouseInventoryItem.getWarehouseId());
        assertEquals(warehouseInventoryItem.getUpdatedAt(), updatedWarehouseInventoryItem.getUpdatedAt());
        assertEquals(warehouseInventoryItem.getQuantity(), updatedWarehouseInventoryItem.getQuantity());
    }

    @Test
    void testDeleteWarehouse() {
        // Arrange
        Optional<WarehouseInventoryItem> warehouseInventoryItemToBeDeletedOptional = warehouseInventoryItemRepository.findById(warehouseInventoryItemId);
        if (warehouseInventoryItemToBeDeletedOptional.isEmpty()) {
            fail("Expected an existing warehouse order with id " + warehouseInventoryItemId);
        }

        WarehouseInventoryItem warehouseInventoryItemToBeDeleted = warehouseInventoryItemToBeDeletedOptional.get();

        // Act
        warehouseInventoryItemRepository.delete(warehouseInventoryItemToBeDeleted);

        // Assert
        Optional<WarehouseInventoryItem> deletedWarehouseInventoryItemOptional = warehouseInventoryItemRepository.findById(warehouseInventoryItemId);
        if (deletedWarehouseInventoryItemOptional.isPresent()) {
            fail("Expected warehouse order with id 1 to have been deleted");
        }
    }

    Warehouse addTestWarehouse() {
        Warehouse warehouse = new Warehouse();
        warehouse.setName("Test Warehouse");
        warehouse.setOrganizationId(organizationId);
        Location location = new Location();
        location.setId(locationId);
        warehouse.setLocation(location);

        return entityManager.persist(warehouse);
    }

    WarehouseInventoryItem addTestWarehouseInventoryItem() {
        WarehouseInventoryItem warehouseInventoryItem = new WarehouseInventoryItem();
        warehouseInventoryItem.setWarehouseId(warehouseId);
        warehouseInventoryItem.setOrganizationId(organizationId);
        warehouseInventoryItem.setQuantity(10f);

        return entityManager.persist(warehouseInventoryItem);
    }
}
