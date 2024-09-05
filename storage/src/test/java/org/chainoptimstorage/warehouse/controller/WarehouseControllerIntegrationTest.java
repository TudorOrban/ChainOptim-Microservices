package org.chainoptimstorage.warehouse.controller;

import org.chainoptimstorage.core.warehouse.model.Warehouse;
import org.chainoptimstorage.internal.in.location.dto.Location;
import org.chainoptimstorage.internal.in.tenant.service.SubscriptionPlanLimiterService;
import org.chainoptimstorage.internal.in.location.dto.CreateLocationDTO;
import org.chainoptimstorage.internal.in.location.service.LocationService;
import org.chainoptimstorage.shared.sanitization.EntitySanitizerService;
import org.chainoptimstorage.shared.PaginatedResults;
import org.chainoptimstorage.core.warehouse.dto.CreateWarehouseDTO;
import org.chainoptimstorage.core.warehouse.dto.WarehousesSearchDTO;
import org.chainoptimstorage.core.warehouse.dto.UpdateWarehouseDTO;
import org.chainoptimstorage.core.warehouse.repository.WarehouseRepository;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
class WarehouseControllerIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    // Seed data services
    @Autowired
    private LocationService locationService;
    @Autowired
    private WarehouseRepository warehouseRepository;
    @Autowired
    private EntitySanitizerService entitySanitizerService;
    @Autowired
    private SubscriptionPlanLimiterService planLimiterService;

    // Necessary seed data
    Integer organizationId = 1;
    String jwtToken = "validToken";
    CreateLocationDTO locationDTO;
    Location location;
    Integer supplierId;

    @BeforeEach
    void setUp() {
        // Set up supplier for search, update and delete tests
        createTestSuppliers();
        
        location = new Location();
        location.setId(1);
    }

    void createTestSuppliers() {
        Warehouse warehouse1 = createTestSupplier("Test Supplier 1");
        supplierId = warehouse1.getId();

        Warehouse warehouse2 = createTestSupplier("Test Supplier 2");

        Warehouse warehouse3 = createTestSupplier("Test Supplier 3");
    }

    Warehouse createTestSupplier(String supplierName) {
        Warehouse warehouse = new Warehouse();
        warehouse.setName(supplierName);
        warehouse.setOrganizationId(organizationId);
        warehouse.setLocation(location);

        return warehouseRepository.save(warehouse);
    }

    @Test
    void testSearchSuppliers() throws Exception {
        // Arrange
        String url = "http://localhost:8080/api/v1/suppliers/organization/advanced/" + organizationId.toString()
                + "?searchQuery=Test"
                + "&sortOption=name"
                + "&ascending=true"
                + "&page=1"
                + "&itemsPerPage=2";
        String invalidJWTToken = "Invalid";

        // Act and assert error status for invalid credentials
        MvcResult invalidMvcResult = mockMvc.perform(get(url)
                .header("Authorization", "Bearer " + invalidJWTToken))
                .andExpect(status().is(403))
                .andReturn();

        // Act
        MvcResult mvcResult = mockMvc.perform(get(url)
                        .header("Authorization", "Bearer " + jwtToken))
                .andExpect(status().isOk())
                .andReturn();

        // Extract and deserialize response
        String responseContent = mvcResult.getResponse().getContentAsString();
        PaginatedResults<WarehousesSearchDTO> paginatedResults = objectMapper.readValue(
                responseContent, new TypeReference<PaginatedResults<WarehousesSearchDTO>>() {});

        // Assert
        assertNotNull(paginatedResults);
        assertEquals(2, paginatedResults.results.size()); // Ensure pagination works
        assertEquals(3, paginatedResults.totalCount); // Ensure total count works
        assertEquals(supplierId, paginatedResults.results.getFirst().getId()); // Ensure sorting works
    }

    @Test
    void testCreateSupplier() throws Exception {
        // Arrange
        CreateWarehouseDTO supplierDTO = new CreateWarehouseDTO("Test Supplier - Unique Title 123456789", organizationId, location.getId(), locationDTO, false);
        String supplierDTOJson = objectMapper.writeValueAsString(supplierDTO);
        String invalidJWTToken = "Invalid";

        // Act (invalid security credentials)
        mockMvc.perform(post("/api/v1/suppliers/create")
                        .header("Authorization", "Bearer " + invalidJWTToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(supplierDTOJson))
                        .andExpect(status().is(403));

        // Assert
        Optional<Warehouse> invalidCreatedSupplierOptional = warehouseRepository.findByName(supplierDTO.getName());
        if (invalidCreatedSupplierOptional.isPresent()) {
            fail("Failed to prevent creation on invalid JWT token");
        }

        // Act
        mockMvc.perform(post("/api/v1/suppliers/create")
                        .header("Authorization", "Bearer " + jwtToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(supplierDTOJson));

        // Assert
        Optional<Warehouse> createdSupplierOptional = warehouseRepository.findByName(supplierDTO.getName());
        if (createdSupplierOptional.isEmpty()) {
            fail("Created supplier has not been found");
        }
        Warehouse createdWarehouse = createdSupplierOptional.get();

        assertNotNull(createdWarehouse);
        assertEquals(supplierDTO.getName(), createdWarehouse.getName());
        assertEquals(supplierDTO.getOrganizationId(), createdWarehouse.getOrganizationId());
        assertEquals(supplierDTO.getLocationId(), createdWarehouse.getLocation().getId());
    }

    @Test
    void testUpdateSupplier() throws Exception {
        // Arrange
        UpdateWarehouseDTO supplierDTO = new UpdateWarehouseDTO(supplierId, "Test Supplier - Updated Unique Title 123456789", organizationId, location.getId(), null, false);
        String supplierDTOJson = objectMapper.writeValueAsString(supplierDTO);
        String invalidJWTToken = "Invalid";

        // Act (invalid security credentials)
        mockMvc.perform(put("/api/v1/suppliers/update")
                        .header("Authorization", "Bearer " + invalidJWTToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(supplierDTOJson))
                        .andExpect(status().is(403));

        // Assert
        Optional<Warehouse> invalidUpdatedSupplierOptional = warehouseRepository.findByName(supplierDTO.getName());
        if (invalidUpdatedSupplierOptional.isPresent()) {
            fail("Failed to prevent update on invalid JWT token.");
        }

        // Act
        mockMvc.perform(put("/api/v1/suppliers/update")
                        .header("Authorization", "Bearer " + jwtToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(supplierDTOJson))
                        .andExpect(status().isOk());

        // Assert
        Optional<Warehouse> updatedSupplierOptional = warehouseRepository.findByName(supplierDTO.getName());
        if (updatedSupplierOptional.isEmpty()) {
            fail("Updated supplier has not been found");
        }
        Warehouse updatedWarehouse = updatedSupplierOptional.get();
        assertNotNull(updatedWarehouse);
        assertEquals(supplierDTO.getName(), updatedWarehouse.getName());
        assertEquals(supplierDTO.getLocationId(), updatedWarehouse.getLocation().getId());
    }

    @Test
    void testDeleteSupplier() throws Exception {
        // Arrange
        String url = "http://localhost:8080/api/v1/suppliers/delete/" + supplierId;
        String invalidJWTToken = "Invalid";

        // Act (invalid security credentials)
        mockMvc.perform(delete(url)
                        .header("Authorization", "Bearer " + invalidJWTToken))
                        .andExpect(status().is(403));

        // Assert
        Optional<Warehouse> invalidUpdatedSupplierOptional = warehouseRepository.findById(supplierId);
        if (invalidUpdatedSupplierOptional.isEmpty()) {
            fail("Failed to prevent deletion on invalid JWT Token.");
        }

        // Act
        mockMvc.perform(delete(url)
                .header("Authorization", "Bearer " + jwtToken))
                .andExpect(status().isOk());

        // Assert
        Optional<Warehouse> updatedSupplierOptional = warehouseRepository.findById(supplierId);
        if (updatedSupplierOptional.isPresent()) {
            fail("Supplier has not been deleted as expected.");
        }
    }

}
