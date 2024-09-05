package org.chainoptimstorage.warehouseinventoryitem.controller;

import org.chainoptimstorage.core.warehouse.model.Warehouse;
import org.chainoptimstorage.core.warehouseinventoryitem.dto.CreateWarehouseInventoryItemDTO;
import org.chainoptimstorage.core.warehouseinventoryitem.dto.UpdateWarehouseInventoryItemDTO;
import org.chainoptimstorage.exception.ResourceNotFoundException;
import org.chainoptimstorage.internal.in.goods.repository.ComponentRepository;
import org.chainoptimstorage.internal.in.goods.model.Component;
import org.chainoptimstorage.internal.in.tenant.service.SubscriptionPlanLimiterService;
import org.chainoptimstorage.shared.PaginatedResults;
import org.chainoptimstorage.shared.sanitization.EntitySanitizerService;
import org.chainoptimstorage.core.warehouse.repository.WarehouseRepository;
import org.chainoptimstorage.core.warehouseinventoryitem.model.WarehouseInventoryItem;
import org.chainoptimstorage.core.warehouseinventoryitem.model.WarehouseInventoryEvent;
import org.chainoptimstorage.core.warehouseinventoryitem.repository.WarehouseInventoryItemRepository;
import org.chainoptimstorage.core.warehouseinventoryitem.service.WarehouseInventoryItemService;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.transaction.Transactional;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
class WarehouseInventoryItemControllerIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private WarehouseInventoryItemService warehouseInventoryItemService;
    @MockBean
    private KafkaTemplate<String, WarehouseInventoryEvent> kafkaTemplate;

    @Autowired
    private ObjectMapper objectMapper;

    // Seed data services
    @Autowired
    private WarehouseInventoryItemRepository warehouseInventoryItemRepository;
    @Autowired
    private WarehouseRepository warehouseRepository;
    @Autowired
    private ComponentRepository componentRepository;
    @Autowired
    private EntitySanitizerService entitySanitizerService;
    @Autowired
    private SubscriptionPlanLimiterService planLimiterService;

    // Necessary seed data
    Integer organizationId = 1;
    String jwtToken = "validToken";
    Integer warehouseId;
    List<Integer> warehouseInventoryItemIds = new ArrayList<>();

    @BeforeEach
    void setUp() {
        createTestWarehouse();

        createTestComponent();

        // Set up warehouse orders for search, update and delete tests
        createTestWarehouseInventoryItems();

        // - Mock the KafkaTemplate send method
        CompletableFuture<SendResult<String, WarehouseInventoryEvent>> completableFuture = CompletableFuture.completedFuture(new SendResult<>(null, null));
        when(kafkaTemplate.send(anyString(), any(WarehouseInventoryEvent.class))).thenReturn(completableFuture);
    }

    @Test
    void testSearchWarehouseInventoryItems() throws Exception {
        // Arrange
        String url = "http://localhost:8080/api/v1/warehouse-orders/organization/advanced/" + warehouseId.toString()
                + "?searchQuery="
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
        PaginatedResults<WarehouseInventoryItem> paginatedResults = objectMapper.readValue(
                responseContent, new TypeReference<PaginatedResults<WarehouseInventoryItem>>() {});

        // Assert
        assertNotNull(paginatedResults);
        assertEquals(2, paginatedResults.results.size()); // Ensure pagination works
        assertEquals(3, paginatedResults.totalCount); // Ensure total count works
        assertEquals(warehouseInventoryItemIds.getFirst(), paginatedResults.results.getFirst().getId()); // Ensure sorting works
    }

    @Test
    void testCreateWarehouseInventoryItem() throws Exception {
        // Arrange
        CreateWarehouseInventoryItemDTO warehouseInventoryItemDTO = getCreateWarehouseInventoryItemDTO("O123");

        String warehouseInventoryItemDTOJson = objectMapper.writeValueAsString(warehouseInventoryItemDTO);
        String invalidJWTToken = "Invalid";

        // Act (invalid security credentials)
        mockMvc.perform(post("/api/v1/warehouse-orders/create")
                        .header("Authorization", "Bearer " + invalidJWTToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(warehouseInventoryItemDTOJson))
                        .andExpect(status().is(403));

        // Assert
        Optional<WarehouseInventoryItem> invalidCreatedWarehouseInventoryItemOptional = warehouseInventoryItemRepository.findByCompanyId(warehouseInventoryItemDTO.getCompanyId());
        if (invalidCreatedWarehouseInventoryItemOptional.isPresent()) {
            fail("Failed to prevent creation on invalid JWT token");
        }

        // Act
        mockMvc.perform(post("/api/v1/warehouse-orders/create")
                .header("Authorization", "Bearer " + jwtToken)
                .contentType(MediaType.APPLICATION_JSON)
                .content(warehouseInventoryItemDTOJson))
                .andExpect(status().isOk());

        // Assert
        Optional<WarehouseInventoryItem> createdWarehouseInventoryItemOptional = warehouseInventoryItemRepository.findByCompanyId(warehouseInventoryItemDTO.getCompanyId());
        if (createdWarehouseInventoryItemOptional.isEmpty()) {
            fail("Created warehouse order has not been found");
        }
        WarehouseInventoryItem createdWarehouseInventoryItem = createdWarehouseInventoryItemOptional.get();

        assertNotNull(createdWarehouseInventoryItem);
        assertEquals(warehouseInventoryItemDTO.getOrganizationId(), createdWarehouseInventoryItem.getOrganizationId());
        assertEquals(warehouseInventoryItemDTO.getWarehouseId(), createdWarehouseInventoryItem.getWarehouseId());
        assertEquals(warehouseInventoryItemDTO.getQuantity(), createdWarehouseInventoryItem.getQuantity());
        assertEquals(warehouseInventoryItemDTO.getComponentId(), createdWarehouseInventoryItem.getComponent().getId());
    }

    @Test
    void testCreateWarehouseInventoryItemsInBulk() throws Exception {
        // Arrange
        List<CreateWarehouseInventoryItemDTO> warehouseInventoryItemDTOs = List.of(
                getCreateWarehouseInventoryItemDTO("O1"),
                getCreateWarehouseInventoryItemDTO("O2"),
                getCreateWarehouseInventoryItemDTO("O3")
        );
        String warehouseInventoryItemDTOJson = objectMapper.writeValueAsString(warehouseInventoryItemDTOs);
        String invalidJWTToken = "Invalid";

        // Act (invalid security credentials)
        mockMvc.perform(post("/api/v1/warehouse-orders/create/bulk")
                        .header("Authorization", "Bearer " + invalidJWTToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(warehouseInventoryItemDTOJson))
                .andExpect(status().is(403));

        // Assert
        Optional<WarehouseInventoryItem> invalidCreatedWarehouseInventoryItemOptional = warehouseInventoryItemRepository.findByCompanyId(warehouseInventoryItemDTOs.getFirst().getCompanyId());
        if (invalidCreatedWarehouseInventoryItemOptional.isPresent()) {
            fail("Failed to prevent creation on invalid JWT token");
        }

        // Act
        mockMvc.perform(post("/api/v1/warehouse-orders/create/bulk")
                        .header("Authorization", "Bearer " + jwtToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(warehouseInventoryItemDTOJson))
                .andExpect(status().isOk());

        // Assert
        List<WarehouseInventoryItem> createdWarehouseInventoryItems = warehouseInventoryItemRepository.findByCompanyIds(warehouseInventoryItemDTOs.stream().map(CreateWarehouseInventoryItemDTO::getCompanyId).toList());
        if (createdWarehouseInventoryItems.size() != 3) {
            fail("Created warehouse order has not been found");
        }
        CreateWarehouseInventoryItemDTO warehouseInventoryItemDTO = warehouseInventoryItemDTOs.getFirst();
        WarehouseInventoryItem createdWarehouseInventoryItem = createdWarehouseInventoryItems.getFirst();

        assertNotNull(createdWarehouseInventoryItems);
        assertEquals(3, createdWarehouseInventoryItems.size());
        assertEquals(warehouseInventoryItemDTO.getCompanyId(), createdWarehouseInventoryItem.getCompanyId());
        assertEquals(warehouseInventoryItemDTO.getOrganizationId(), createdWarehouseInventoryItem.getOrganizationId());
        assertEquals(warehouseInventoryItemDTO.getWarehouseId(), createdWarehouseInventoryItem.getWarehouseId());
        assertEquals(warehouseInventoryItemDTO.getQuantity(), createdWarehouseInventoryItem.getQuantity());
        assertEquals(warehouseInventoryItemDTO.getComponentId(), createdWarehouseInventoryItem.getComponent().getId());
    }

    @Test
    @Transactional
    void testUpdateWarehouseInventoryItemsInBulk() throws Exception {
        // Arrange
        List<WarehouseInventoryItem> existingOrders = warehouseInventoryItemRepository.findByIds(warehouseInventoryItemIds)
                .orElseThrow(() -> new ResourceNotFoundException("Warehouse orders not found"));
        List<UpdateWarehouseInventoryItemDTO> warehouseInventoryItemDTOs = existingOrders.stream()
                .map(this::getUpdateWarehouseInventoryItemDTO).toList();

        String warehouseInventoryItemDTOJson = objectMapper.writeValueAsString(warehouseInventoryItemDTOs);
        String invalidJWTToken = "Invalid";
        List<String> updatedCompanyIds = warehouseInventoryItemDTOs.stream().map(UpdateWarehouseInventoryItemDTO::getCompanyId).toList();

        // Act (invalid security credentials)
        mockMvc.perform(put("/api/v1/warehouse-orders/update/bulk")
                        .header("Authorization", "Bearer " + invalidJWTToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(warehouseInventoryItemDTOJson))
                .andExpect(status().is(403));

        // Assert
        List<WarehouseInventoryItem> invalidUpdatedWarehouseInventoryItems = warehouseInventoryItemRepository.findByCompanyIds(updatedCompanyIds);
        if (!invalidUpdatedWarehouseInventoryItems.isEmpty()) {
            fail("Failed to prevent update on invalid JWT token");
        }

        // Act
        mockMvc.perform(put("/api/v1/warehouse-orders/update/bulk")
                        .header("Authorization", "Bearer " + jwtToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(warehouseInventoryItemDTOJson))
                .andExpect(status().isOk()).andReturn();

        // Assert
        List<WarehouseInventoryItem> updatedWarehouseInventoryItems = warehouseInventoryItemRepository.findByCompanyIds(updatedCompanyIds);

        if (updatedWarehouseInventoryItems.size() != 3) {
            fail("Updated warehouse orders have not been found");
        }
        UpdateWarehouseInventoryItemDTO warehouseInventoryItemDTO = warehouseInventoryItemDTOs.getFirst();
        WarehouseInventoryItem createdWarehouseInventoryItem = updatedWarehouseInventoryItems.getFirst();

        assertNotNull(updatedWarehouseInventoryItems);
        assertEquals(3, updatedWarehouseInventoryItems.size());
        assertEquals(warehouseInventoryItemDTO.getCompanyId(), createdWarehouseInventoryItem.getCompanyId());
        assertEquals(warehouseInventoryItemDTO.getOrganizationId(), createdWarehouseInventoryItem.getOrganizationId());
        assertEquals(warehouseInventoryItemDTO.getQuantity(), createdWarehouseInventoryItem.getQuantity());
        assertEquals(warehouseInventoryItemDTO.getComponentId(), createdWarehouseInventoryItem.getComponent().getId());
    }

    @Test
    void testDeleteWarehouseInventoryItemsInBulk() throws Exception {
        // Arrange
        String orderIdsJson = objectMapper.writeValueAsString(warehouseInventoryItemIds);
        String invalidJWTToken = "Invalid";

        // Act (invalid security credentials)
        mockMvc.perform(delete("/api/v1/warehouse-orders/delete/bulk")
                        .header("Authorization", "Bearer " + invalidJWTToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(orderIdsJson))
                .andExpect(status().is(403));

        // Assert
        List<WarehouseInventoryItem> invalidDeletedWarehouseInventoryItems = warehouseInventoryItemRepository.findByIds(warehouseInventoryItemIds)
                .orElseThrow(() -> new ResourceNotFoundException("Warehouse orders not found"));
        if (invalidDeletedWarehouseInventoryItems.size() != 3) {
            fail("Failed to prevent deletion on invalid JWT token");
        }

        // Act
        mockMvc.perform(delete("/api/v1/warehouse-orders/delete/bulk")
                        .header("Authorization", "Bearer " + jwtToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(orderIdsJson))
                .andExpect(status().isOk());

        // Assert
        Optional<List<WarehouseInventoryItem>> deletedWarehouseInventoryItems = warehouseInventoryItemRepository.findByIds(warehouseInventoryItemIds);
        if (deletedWarehouseInventoryItems.isPresent() && !deletedWarehouseInventoryItems.get().isEmpty()) {
            fail("Deleted warehouse orders have been found");
        }
    }

    private CreateWarehouseInventoryItemDTO getCreateWarehouseInventoryItemDTO(String companyId) {
        CreateWarehouseInventoryItemDTO warehouseInventoryItemDTO = new CreateWarehouseInventoryItemDTO();
        warehouseInventoryItemDTO.setOrganizationId(organizationId);
        warehouseInventoryItemDTO.setWarehouseId(warehouseId);
        warehouseInventoryItemDTO.setCompanyId(companyId);
        warehouseInventoryItemDTO.setQuantity(10f);
        return warehouseInventoryItemDTO;
    }

    private UpdateWarehouseInventoryItemDTO getUpdateWarehouseInventoryItemDTO(WarehouseInventoryItem order) {
        UpdateWarehouseInventoryItemDTO updateWarehouseInventoryItemDTO = new UpdateWarehouseInventoryItemDTO();
        updateWarehouseInventoryItemDTO.setId(order.getId());
        updateWarehouseInventoryItemDTO.setCompanyId(order.getCompanyId() + " Updated");
        updateWarehouseInventoryItemDTO.setOrganizationId(order.getOrganizationId());
        updateWarehouseInventoryItemDTO.setComponentId(order.getComponent().getId());
        updateWarehouseInventoryItemDTO.setQuantity(order.getQuantity() != null ? order.getQuantity() + 10f : null);

        return updateWarehouseInventoryItemDTO;
    }

    // Testing
    void createTestWarehouse() {
        Warehouse warehouse = new Warehouse();
        warehouse.setOrganizationId(organizationId);
        warehouse.setName("Test Warehouse");

        warehouseId = warehouseRepository.save(warehouse).getId();
    }

    void createTestComponent() {
        Component newComponent = new Component();
        newComponent.setOrganizationId(organizationId);
        newComponent.setName("Test Component");

//        component = componentRepository.save(newComponent);
    }

    void createTestWarehouseInventoryItems() {
        WarehouseInventoryItem warehouseInventoryItem1 = createTestWarehouseInventoryItem("O01");
        warehouseInventoryItemIds.add(warehouseInventoryItem1.getId());

        WarehouseInventoryItem warehouseInventoryItem2 = createTestWarehouseInventoryItem("O02");
        warehouseInventoryItemIds.add(warehouseInventoryItem2.getId());

        WarehouseInventoryItem warehouseInventoryItem3 = createTestWarehouseInventoryItem("O03");
        warehouseInventoryItemIds.add(warehouseInventoryItem3.getId());
    }

    WarehouseInventoryItem createTestWarehouseInventoryItem(String companyId) {
        WarehouseInventoryItem warehouseInventoryItem = new WarehouseInventoryItem();
        warehouseInventoryItem.setCompanyId(companyId);
        warehouseInventoryItem.setOrganizationId(organizationId);
        warehouseInventoryItem.setWarehouseId(warehouseId);

        return warehouseInventoryItemRepository.save(warehouseInventoryItem);
    }
}
