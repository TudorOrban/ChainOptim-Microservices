package org.chainoptimstorage.warehouse.repository;

import org.chainoptimstorage.core.warehouse.model.Warehouse;
import org.chainoptimstorage.core.warehouse.repository.WarehouseRepository;
import org.chainoptimstorage.internal.in.location.dto.Location;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.boot.test.autoconfigure.orm.jpa.TestEntityManager;
import org.springframework.test.context.junit.jupiter.SpringExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase.Replace.NONE;

@ExtendWith(SpringExtension.class)
@DataJpaTest
@AutoConfigureTestDatabase(replace = NONE)
class WarehouseRepositoryTest {

    @Autowired
    private TestEntityManager entityManager;

    @Autowired
    private WarehouseRepository warehouseRepository;

    Integer organizationId = 1;
    Location location;
    Integer supplierId;

    @BeforeEach
    void setUp() {
        // Set up supplier for update and delete tests
        location = new Location();
        location.setId(1);
        Warehouse warehouse = addTestWarehouse();
        supplierId = warehouse.getId();
    }

    @Test
    void testCreateWarehouse() {
        // Arrange
        Warehouse savedWarehouse = addTestWarehouse();

        entityManager.flush();
        entityManager.clear();

        // Assert
        Optional<Warehouse> foundWarehouseOpt = warehouseRepository.findById(savedWarehouse.getId());
        assertTrue(foundWarehouseOpt.isPresent(), "Warehouse should be found in the database");

        Warehouse foundWarehouse = foundWarehouseOpt.get();
        assertEquals(savedWarehouse.getName(), foundWarehouse.getName());
        assertEquals(savedWarehouse.getOrganizationId(), foundWarehouse.getOrganizationId());
        assertEquals(savedWarehouse.getLocation(), foundWarehouse.getLocation());
    }

    @Test
    void testUpdateWarehouse() {
        // Arrange
        Optional<Warehouse> supplierOptional = warehouseRepository.findById(supplierId); // Id from setUp
        if (supplierOptional.isEmpty()) {
            fail("Expected an existing supplier with id " + supplierOptional);
        }

        Warehouse warehouse = supplierOptional.get();
        warehouse.setName("New Test Name");

        // Act
        Warehouse updatedWarehouse = warehouseRepository.save(warehouse);

        // Assert
        assertNotNull(updatedWarehouse);
        assertEquals("New Test Name", updatedWarehouse.getName());
    }

    @Test
    void testDeleteWarehouse() {
        // Arrange
        Optional<Warehouse> supplierToBeDeletedOptional = warehouseRepository.findById(supplierId);
        if (supplierToBeDeletedOptional.isEmpty()) {
            fail("Expected an existing supplier with id " + supplierId);
        }

        Warehouse warehouseToBeDeleted = supplierToBeDeletedOptional.get();

        // Act
        warehouseRepository.delete(warehouseToBeDeleted);

        // Assert
        Optional<Warehouse> deletedWarehouseOptional = warehouseRepository.findById(supplierId);
        if (deletedWarehouseOptional.isPresent()) {
            fail("Expected supplier with id 1 to have been deleted");
        }
    }

    Warehouse addTestWarehouse() {
        Warehouse warehouse = new Warehouse();
        warehouse.setName("Test Warehouse");
        warehouse.setOrganizationId(organizationId);
        warehouse.setLocation(location);

        return entityManager.persist(warehouse);
    }
}
