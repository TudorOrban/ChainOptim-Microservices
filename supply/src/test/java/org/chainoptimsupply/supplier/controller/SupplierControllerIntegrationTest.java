package org.chainoptimsupply.supplier.controller;

import org.chainoptimsupply.internal.in.tenant.service.SubscriptionPlanLimiterService;
import org.chainoptimsupply.internal.in.location.dto.CreateLocationDTO;
import org.chainoptimsupply.internal.in.location.service.LocationService;
import org.chainoptimsupply.shared.sanitization.EntitySanitizerService;
import org.chainoptimsupply.shared.PaginatedResults;
import org.chainoptimsupply.core.supplier.dto.CreateSupplierDTO;
import org.chainoptimsupply.core.supplier.dto.SuppliersSearchDTO;
import org.chainoptimsupply.core.supplier.dto.UpdateSupplierDTO;
import org.chainoptimsupply.core.supplier.model.Supplier;
import org.chainoptimsupply.core.supplier.repository.SupplierRepository;

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
class SupplierControllerIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    // Seed data services
    @Autowired
    private LocationService locationService;
    @Autowired
    private SupplierRepository supplierRepository;
    @Autowired
    private EntitySanitizerService entitySanitizerService;
    @Autowired
    private SubscriptionPlanLimiterService planLimiterService;

    // Necessary seed data
    Integer organizationId = 1;
    String jwtToken = "validToken";
    CreateLocationDTO locationDTO;
    Integer locationId = 1;
    Integer supplierId;

    @BeforeEach
    void setUp() {
        // Set up supplier for search, update and delete tests
        createTestSuppliers();
    }

    void createTestSuppliers() {
        Supplier supplier1 = createTestSupplier("Test Supplier 1");
        supplierId = supplier1.getId();

        Supplier supplier2 = createTestSupplier("Test Supplier 2");

        Supplier supplier3 = createTestSupplier("Test Supplier 3");
    }

    Supplier createTestSupplier(String supplierName) {
        Supplier supplier = new Supplier();
        supplier.setName(supplierName);
        supplier.setOrganizationId(organizationId);
        supplier.setLocationId(locationId);

        return supplierRepository.save(supplier);
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
        PaginatedResults<SuppliersSearchDTO> paginatedResults = objectMapper.readValue(
                responseContent, new TypeReference<PaginatedResults<SuppliersSearchDTO>>() {});

        // Assert
        assertNotNull(paginatedResults);
        assertEquals(2, paginatedResults.results.size()); // Ensure pagination works
        assertEquals(3, paginatedResults.totalCount); // Ensure total count works
        assertEquals(supplierId, paginatedResults.results.getFirst().getId()); // Ensure sorting works
    }

    @Test
    void testCreateSupplier() throws Exception {
        // Arrange
        CreateSupplierDTO supplierDTO = new CreateSupplierDTO("Test Supplier - Unique Title 123456789", organizationId, locationId, locationDTO, false);
        String supplierDTOJson = objectMapper.writeValueAsString(supplierDTO);
        String invalidJWTToken = "Invalid";

        // Act (invalid security credentials)
        mockMvc.perform(post("/api/v1/suppliers/create")
                        .header("Authorization", "Bearer " + invalidJWTToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(supplierDTOJson))
                        .andExpect(status().is(403));

        // Assert
        Optional<Supplier> invalidCreatedSupplierOptional = supplierRepository.findByName(supplierDTO.getName());
        if (invalidCreatedSupplierOptional.isPresent()) {
            fail("Failed to prevent creation on invalid JWT token");
        }

        // Act
        mockMvc.perform(post("/api/v1/suppliers/create")
                        .header("Authorization", "Bearer " + jwtToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(supplierDTOJson));

        // Assert
        Optional<Supplier> createdSupplierOptional = supplierRepository.findByName(supplierDTO.getName());
        if (createdSupplierOptional.isEmpty()) {
            fail("Created supplier has not been found");
        }
        Supplier createdSupplier = createdSupplierOptional.get();

        assertNotNull(createdSupplier);
        assertEquals(supplierDTO.getName(), createdSupplier.getName());
        assertEquals(supplierDTO.getOrganizationId(), createdSupplier.getOrganizationId());
        assertEquals(supplierDTO.getLocationId(), createdSupplier.getLocationId());
    }

    @Test
    void testUpdateSupplier() throws Exception {
        // Arrange
        UpdateSupplierDTO supplierDTO = new UpdateSupplierDTO(supplierId, "Test Supplier - Updated Unique Title 123456789", locationId, null, false);
        String supplierDTOJson = objectMapper.writeValueAsString(supplierDTO);
        String invalidJWTToken = "Invalid";

        // Act (invalid security credentials)
        mockMvc.perform(put("/api/v1/suppliers/update")
                        .header("Authorization", "Bearer " + invalidJWTToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(supplierDTOJson))
                        .andExpect(status().is(403));

        // Assert
        Optional<Supplier> invalidUpdatedSupplierOptional = supplierRepository.findByName(supplierDTO.getName());
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
        Optional<Supplier> updatedSupplierOptional = supplierRepository.findByName(supplierDTO.getName());
        if (updatedSupplierOptional.isEmpty()) {
            fail("Updated supplier has not been found");
        }
        Supplier updatedSupplier = updatedSupplierOptional.get();
        assertNotNull(updatedSupplier);
        assertEquals(supplierDTO.getName(), updatedSupplier.getName());
        assertEquals(supplierDTO.getLocationId(), updatedSupplier.getLocationId());
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
        Optional<Supplier> invalidUpdatedSupplierOptional = supplierRepository.findById(supplierId);
        if (invalidUpdatedSupplierOptional.isEmpty()) {
            fail("Failed to prevent deletion on invalid JWT Token.");
        }

        // Act
        mockMvc.perform(delete(url)
                .header("Authorization", "Bearer " + jwtToken))
                .andExpect(status().isOk());

        // Assert
        Optional<Supplier> updatedSupplierOptional = supplierRepository.findById(supplierId);
        if (updatedSupplierOptional.isPresent()) {
            fail("Supplier has not been deleted as expected.");
        }
    }

}
