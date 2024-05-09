package org.chainoptimdemand.clientorder.controller;

import org.chainoptimdemand.exception.ResourceNotFoundException;
import org.chainoptimdemand.internal.in.goods.repository.ComponentRepository;
import org.chainoptimdemand.internal.in.goods.model.Component;
import org.chainoptimdemand.internal.in.tenant.service.SubscriptionPlanLimiterService;
import org.chainoptimdemand.shared.PaginatedResults;
import org.chainoptimdemand.shared.sanitization.EntitySanitizerService;
import org.chainoptimdemand.core.client.dto.CreateClientOrderDTO;
import org.chainoptimdemand.core.client.dto.UpdateClientOrderDTO;
import org.chainoptimdemand.core.client.repository.ClientRepository;
import org.chainoptimdemand.core.clientorder.model.OrderStatus;
import org.chainoptimdemand.core.clientorder.model.ClientOrder;
import org.chainoptimdemand.core.client.model.Client;
import org.chainoptimdemand.core.clientorder.model.ClientOrderEvent;
import org.chainoptimdemand.core.clientorder.repository.ClientOrderRepository;
import org.chainoptimdemand.core.clientorder.service.ClientOrderService;

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
class ClientOrderControllerIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ClientOrderService clientOrderService;
    @MockBean
    private KafkaTemplate<String, ClientOrderEvent> kafkaTemplate;

    @Autowired
    private ObjectMapper objectMapper;

    // Seed data services
    @Autowired
    private ClientOrderRepository clientOrderRepository;
    @Autowired
    private ClientRepository clientRepository;
    @Autowired
    private ComponentRepository componentRepository;
    @Autowired
    private EntitySanitizerService entitySanitizerService;
    @Autowired
    private SubscriptionPlanLimiterService planLimiterService;

    // Necessary seed data
    Integer organizationId = 1;
    String jwtToken = "validToken";
    Integer clientId;
    List<Integer> clientOrderIds = new ArrayList<>();
    Integer componentId = 1;

    @BeforeEach
    void setUp() {
        createTestClient();

        createTestComponent();

        // Set up client orders for search, update and delete tests
        createTestClientOrders();

        // - Mock the KafkaTemplate send method
        CompletableFuture<SendResult<String, ClientOrderEvent>> completableFuture = CompletableFuture.completedFuture(new SendResult<>(null, null));
        when(kafkaTemplate.send(anyString(), any(ClientOrderEvent.class))).thenReturn(completableFuture);
    }

    @Test
    void testSearchClientOrders() throws Exception {
        // Arrange
        String url = "http://localhost:8080/api/v1/client-orders/organization/advanced/" + clientId.toString()
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
        PaginatedResults<ClientOrder> paginatedResults = objectMapper.readValue(
                responseContent, new TypeReference<PaginatedResults<ClientOrder>>() {});

        // Assert
        assertNotNull(paginatedResults);
        assertEquals(2, paginatedResults.results.size()); // Ensure pagination works
        assertEquals(3, paginatedResults.totalCount); // Ensure total count works
        assertEquals(clientOrderIds.getFirst(), paginatedResults.results.getFirst().getId()); // Ensure sorting works
    }

    @Test
    void testCreateClientOrder() throws Exception {
        // Arrange
        CreateClientOrderDTO clientOrderDTO = getCreateClientOrderDTO("O123");

        String clientOrderDTOJson = objectMapper.writeValueAsString(clientOrderDTO);
        String invalidJWTToken = "Invalid";

        // Act (invalid security credentials)
        mockMvc.perform(post("/api/v1/client-orders/create")
                        .header("Authorization", "Bearer " + invalidJWTToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(clientOrderDTOJson))
                        .andExpect(status().is(403));

        // Assert
        Optional<ClientOrder> invalidCreatedClientOrderOptional = clientOrderRepository.findByCompanyId(clientOrderDTO.getCompanyId());
        if (invalidCreatedClientOrderOptional.isPresent()) {
            fail("Failed to prevent creation on invalid JWT token");
        }

        // Act
        mockMvc.perform(post("/api/v1/client-orders/create")
                .header("Authorization", "Bearer " + jwtToken)
                .contentType(MediaType.APPLICATION_JSON)
                .content(clientOrderDTOJson))
                .andExpect(status().isOk());

        // Assert
        Optional<ClientOrder> createdClientOrderOptional = clientOrderRepository.findByCompanyId(clientOrderDTO.getCompanyId());
        if (createdClientOrderOptional.isEmpty()) {
            fail("Created client order has not been found");
        }
        ClientOrder createdClientOrder = createdClientOrderOptional.get();

        assertNotNull(createdClientOrder);
        assertEquals(clientOrderDTO.getCompanyId(), createdClientOrder.getCompanyId());
        assertEquals(clientOrderDTO.getOrganizationId(), createdClientOrder.getOrganizationId());
        assertEquals(clientOrderDTO.getClientId(), createdClientOrder.getClientId());
        assertEquals(clientOrderDTO.getQuantity(), createdClientOrder.getQuantity());
        assertEquals(clientOrderDTO.getStatus(), createdClientOrder.getStatus());
        assertEquals(clientOrderDTO.getComponentId(), createdClientOrder.getComponentId());
    }

    @Test
    void testCreateClientOrdersInBulk() throws Exception {
        // Arrange
        List<CreateClientOrderDTO> clientOrderDTOs = List.of(
                getCreateClientOrderDTO("O1"),
                getCreateClientOrderDTO("O2"),
                getCreateClientOrderDTO("O3")
        );
        String clientOrderDTOJson = objectMapper.writeValueAsString(clientOrderDTOs);
        String invalidJWTToken = "Invalid";

        // Act (invalid security credentials)
        mockMvc.perform(post("/api/v1/client-orders/create/bulk")
                        .header("Authorization", "Bearer " + invalidJWTToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(clientOrderDTOJson))
                .andExpect(status().is(403));

        // Assert
        Optional<ClientOrder> invalidCreatedClientOrderOptional = clientOrderRepository.findByCompanyId(clientOrderDTOs.getFirst().getCompanyId());
        if (invalidCreatedClientOrderOptional.isPresent()) {
            fail("Failed to prevent creation on invalid JWT token");
        }

        // Act
        mockMvc.perform(post("/api/v1/client-orders/create/bulk")
                        .header("Authorization", "Bearer " + jwtToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(clientOrderDTOJson))
                .andExpect(status().isOk());

        // Assert
        List<ClientOrder> createdClientOrders = clientOrderRepository.findByCompanyIds(clientOrderDTOs.stream().map(CreateClientOrderDTO::getCompanyId).toList());
        if (createdClientOrders.size() != 3) {
            fail("Created client order has not been found");
        }
        CreateClientOrderDTO clientOrderDTO = clientOrderDTOs.getFirst();
        ClientOrder createdClientOrder = createdClientOrders.getFirst();

        assertNotNull(createdClientOrders);
        assertEquals(3, createdClientOrders.size());
        assertEquals(clientOrderDTO.getCompanyId(), createdClientOrder.getCompanyId());
        assertEquals(clientOrderDTO.getOrganizationId(), createdClientOrder.getOrganizationId());
        assertEquals(clientOrderDTO.getClientId(), createdClientOrder.getClientId());
        assertEquals(clientOrderDTO.getQuantity(), createdClientOrder.getQuantity());
        assertEquals(clientOrderDTO.getStatus(), createdClientOrder.getStatus());
        assertEquals(clientOrderDTO.getComponentId(), createdClientOrder.getComponentId());
    }

    @Test
    @Transactional
    void testUpdateClientOrdersInBulk() throws Exception {
        // Arrange
        List<ClientOrder> existingOrders = clientOrderRepository.findByIds(clientOrderIds)
                .orElseThrow(() -> new ResourceNotFoundException("Client orders not found"));
        List<UpdateClientOrderDTO> clientOrderDTOs = existingOrders.stream()
                .map(this::getUpdateClientOrderDTO).toList();

        String clientOrderDTOJson = objectMapper.writeValueAsString(clientOrderDTOs);
        String invalidJWTToken = "Invalid";
        List<String> updatedCompanyIds = clientOrderDTOs.stream().map(UpdateClientOrderDTO::getCompanyId).toList();

        // Act (invalid security credentials)
        mockMvc.perform(put("/api/v1/client-orders/update/bulk")
                        .header("Authorization", "Bearer " + invalidJWTToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(clientOrderDTOJson))
                .andExpect(status().is(403));

        // Assert
        List<ClientOrder> invalidUpdatedClientOrders = clientOrderRepository.findByCompanyIds(updatedCompanyIds);
        if (!invalidUpdatedClientOrders.isEmpty()) {
            fail("Failed to prevent update on invalid JWT token");
        }

        // Act
        mockMvc.perform(put("/api/v1/client-orders/update/bulk")
                        .header("Authorization", "Bearer " + jwtToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(clientOrderDTOJson))
                .andExpect(status().isOk()).andReturn();

        // Assert
        List<ClientOrder> updatedClientOrders = clientOrderRepository.findByCompanyIds(updatedCompanyIds);

        if (updatedClientOrders.size() != 3) {
            fail("Updated client orders have not been found");
        }
        UpdateClientOrderDTO clientOrderDTO = clientOrderDTOs.getFirst();
        ClientOrder createdClientOrder = updatedClientOrders.getFirst();

        assertNotNull(updatedClientOrders);
        assertEquals(3, updatedClientOrders.size());
        assertEquals(clientOrderDTO.getCompanyId(), createdClientOrder.getCompanyId());
        assertEquals(clientOrderDTO.getOrganizationId(), createdClientOrder.getOrganizationId());
        assertEquals(clientOrderDTO.getQuantity(), createdClientOrder.getQuantity());
        assertEquals(clientOrderDTO.getStatus(), createdClientOrder.getStatus());
        assertEquals(clientOrderDTO.getComponentId(), createdClientOrder.getComponentId());
    }

    @Test
    void testDeleteClientOrdersInBulk() throws Exception {
        // Arrange
        String orderIdsJson = objectMapper.writeValueAsString(clientOrderIds);
        String invalidJWTToken = "Invalid";

        // Act (invalid security credentials)
        mockMvc.perform(delete("/api/v1/client-orders/delete/bulk")
                        .header("Authorization", "Bearer " + invalidJWTToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(orderIdsJson))
                .andExpect(status().is(403));

        // Assert
        List<ClientOrder> invalidDeletedClientOrders = clientOrderRepository.findByIds(clientOrderIds)
                .orElseThrow(() -> new ResourceNotFoundException("Client orders not found"));
        if (invalidDeletedClientOrders.size() != 3) {
            fail("Failed to prevent deletion on invalid JWT token");
        }

        // Act
        mockMvc.perform(delete("/api/v1/client-orders/delete/bulk")
                        .header("Authorization", "Bearer " + jwtToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(orderIdsJson))
                .andExpect(status().isOk());

        // Assert
        Optional<List<ClientOrder>> deletedClientOrders = clientOrderRepository.findByIds(clientOrderIds);
        if (deletedClientOrders.isPresent() && !deletedClientOrders.get().isEmpty()) {
            fail("Deleted client orders have been found");
        }
    }

    private CreateClientOrderDTO getCreateClientOrderDTO(String companyId) {
        CreateClientOrderDTO clientOrderDTO = new CreateClientOrderDTO();
        clientOrderDTO.setOrganizationId(organizationId);
        clientOrderDTO.setClientId(clientId);
        clientOrderDTO.setCompanyId(companyId);
        clientOrderDTO.setComponentId(componentId);
        clientOrderDTO.setQuantity(10f);
        clientOrderDTO.setStatus(OrderStatus.PLACED);
        return clientOrderDTO;
    }

    private UpdateClientOrderDTO getUpdateClientOrderDTO(ClientOrder order) {
        UpdateClientOrderDTO updateClientOrderDTO = new UpdateClientOrderDTO();
        updateClientOrderDTO.setId(order.getId());
        updateClientOrderDTO.setCompanyId(order.getCompanyId() + " Updated");
        updateClientOrderDTO.setOrganizationId(order.getOrganizationId());
        updateClientOrderDTO.setComponentId(order.getComponentId());
        updateClientOrderDTO.setQuantity(order.getQuantity() != null ? order.getQuantity() + 10f : null);
        updateClientOrderDTO.setStatus(OrderStatus.DELIVERED);

        return updateClientOrderDTO;
    }

    // Testing
    void createTestClient() {
        Client client = new Client();
        client.setOrganizationId(organizationId);
        client.setName("Test Client");

        clientId = clientRepository.save(client).getId();
    }

    void createTestComponent() {
        Component newComponent = new Component();
        newComponent.setOrganizationId(organizationId);
        newComponent.setName("Test Component");

//        component = componentRepository.save(newComponent);
    }

    void createTestClientOrders() {
        ClientOrder clientOrder1 = createTestClientOrder("O01");
        clientOrderIds.add(clientOrder1.getId());

        ClientOrder clientOrder2 = createTestClientOrder("O02");
        clientOrderIds.add(clientOrder2.getId());

        ClientOrder clientOrder3 = createTestClientOrder("O03");
        clientOrderIds.add(clientOrder3.getId());
    }

    ClientOrder createTestClientOrder(String companyId) {
        ClientOrder clientOrder = new ClientOrder();
        clientOrder.setCompanyId(companyId);
        clientOrder.setOrganizationId(organizationId);
        clientOrder.setClientId(clientId);
        clientOrder.setComponentId(componentId);
        clientOrder.setStatus(OrderStatus.PLACED);

        return clientOrderRepository.save(clientOrder);
    }
}
