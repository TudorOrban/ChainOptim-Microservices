package org.chainoptimsupply.supplier.repository;

import org.chainoptimsupply.shared.dto.Location;
import org.chainoptimsupply.supplier.model.Supplier;
import org.chainoptimsupply.tenant.model.Organization;
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
class SupplierRepositoryTest {

    @Autowired
    private TestEntityManager entityManager;

    @Autowired
    private SupplierRepository supplierRepository;

    Integer organizationId = 1;
    Integer locationId = 1;
    Integer supplierId;

    @BeforeEach
    void setUp() {
        // Set up supplier for update and delete tests
        Supplier supplier = addTestSupplier();
        supplierId = supplier.getId();
    }

    @Test
    void testCreateSupplier() {
        // Arrange
        Supplier savedSupplier = addTestSupplier();

        entityManager.flush();
        entityManager.clear();

        // Assert
        Optional<Supplier> foundSupplierOpt = supplierRepository.findById(savedSupplier.getId());
        assertTrue(foundSupplierOpt.isPresent(), "Supplier should be found in the database");

        Supplier foundSupplier = foundSupplierOpt.get();
        assertEquals(savedSupplier.getName(), foundSupplier.getName());
        assertEquals(savedSupplier.getOrganizationId(), foundSupplier.getOrganizationId());
        assertEquals(savedSupplier.getLocationId(), foundSupplier.getLocationId());
    }

    @Test
    void testUpdateSupplier() {
        // Arrange
        Optional<Supplier> supplierOptional = supplierRepository.findById(supplierId); // Id from setUp
        if (supplierOptional.isEmpty()) {
            fail("Expected an existing supplier with id " + supplierOptional);
        }

        Supplier supplier = supplierOptional.get();
        supplier.setName("New Test Name");

        // Act
        Supplier updatedSupplier = supplierRepository.save(supplier);

        // Assert
        assertNotNull(updatedSupplier);
        assertEquals("New Test Name", updatedSupplier.getName());
    }

    @Test
    void testDeleteSupplier() {
        // Arrange
        Optional<Supplier> supplierToBeDeletedOptional = supplierRepository.findById(supplierId);
        if (supplierToBeDeletedOptional.isEmpty()) {
            fail("Expected an existing supplier with id " + supplierId);
        }

        Supplier supplierToBeDeleted = supplierToBeDeletedOptional.get();

        // Act
        supplierRepository.delete(supplierToBeDeleted);

        // Assert
        Optional<Supplier> deletedSupplierOptional = supplierRepository.findById(supplierId);
        if (deletedSupplierOptional.isPresent()) {
            fail("Expected supplier with id 1 to have been deleted");
        }
    }

    Supplier addTestSupplier() {
        Supplier supplier = new Supplier();
        supplier.setName("Test Supplier");
        supplier.setOrganizationId(organizationId);
        supplier.setLocationId(locationId);

        return entityManager.persist(supplier);
    }
}
