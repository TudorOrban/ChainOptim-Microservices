package org.chainoptimdemand.client.controller;

import org.chainoptimdemand.internal.in.tenant.service.SubscriptionPlanLimiterService;
import org.chainoptimdemand.internal.in.location.dto.CreateLocationDTO;
import org.chainoptimdemand.internal.in.location.service.LocationService;
import org.chainoptimdemand.shared.sanitization.EntitySanitizerService;
import org.chainoptimdemand.shared.PaginatedResults;
import org.chainoptimdemand.core.client.dto.CreateClientDTO;
import org.chainoptimdemand.core.client.dto.ClientsSearchDTO;
import org.chainoptimdemand.core.client.dto.UpdateClientDTO;
import org.chainoptimdemand.core.client.model.Client;
import org.chainoptimdemand.core.client.repository.ClientRepository;

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
class ClientControllerIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    // Seed data services
    @Autowired
    private LocationService locationService;
    @Autowired
    private ClientRepository clientRepository;
    @Autowired
    private EntitySanitizerService entitySanitizerService;
    @Autowired
    private SubscriptionPlanLimiterService planLimiterService;

    // Necessary seed data
    Integer organizationId = 1;
    String jwtToken = "validToken";
    CreateLocationDTO locationDTO;
    Integer locationId = 1;
    Integer clientId;

    @BeforeEach
    void setUp() {
        // Set up client for search, update and delete tests
        createTestClients();
    }

    void createTestClients() {
        Client client1 = createTestClient("Test Client 1");
        clientId = client1.getId();

        Client client2 = createTestClient("Test Client 2");

        Client client3 = createTestClient("Test Client 3");
    }

    Client createTestClient(String clientName) {
        Client client = new Client();
        client.setName(clientName);
        client.setOrganizationId(organizationId);
        client.setLocationId(locationId);

        return clientRepository.save(client);
    }

    @Test
    void testSearchClients() throws Exception {
        // Arrange
        String url = "http://localhost:8080/api/v1/clients/organization/advanced/" + organizationId.toString()
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
        PaginatedResults<ClientsSearchDTO> paginatedResults = objectMapper.readValue(
                responseContent, new TypeReference<PaginatedResults<ClientsSearchDTO>>() {});

        // Assert
        assertNotNull(paginatedResults);
        assertEquals(2, paginatedResults.results.size()); // Ensure pagination works
        assertEquals(3, paginatedResults.totalCount); // Ensure total count works
        assertEquals(clientId, paginatedResults.results.getFirst().getId()); // Ensure sorting works
    }

    @Test
    void testCreateClient() throws Exception {
        // Arrange
        CreateClientDTO clientDTO = new CreateClientDTO("Test Client - Unique Title 123456789", organizationId, locationId, locationDTO, false);
        String clientDTOJson = objectMapper.writeValueAsString(clientDTO);
        String invalidJWTToken = "Invalid";

        // Act (invalid security credentials)
        mockMvc.perform(post("/api/v1/clients/create")
                        .header("Authorization", "Bearer " + invalidJWTToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(clientDTOJson))
                        .andExpect(status().is(403));

        // Assert
        Optional<Client> invalidCreatedClientOptional = clientRepository.findByName(clientDTO.getName());
        if (invalidCreatedClientOptional.isPresent()) {
            fail("Failed to prevent creation on invalid JWT token");
        }

        // Act
        mockMvc.perform(post("/api/v1/clients/create")
                        .header("Authorization", "Bearer " + jwtToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(clientDTOJson));

        // Assert
        Optional<Client> createdClientOptional = clientRepository.findByName(clientDTO.getName());
        if (createdClientOptional.isEmpty()) {
            fail("Created client has not been found");
        }
        Client createdClient = createdClientOptional.get();

        assertNotNull(createdClient);
        assertEquals(clientDTO.getName(), createdClient.getName());
        assertEquals(clientDTO.getOrganizationId(), createdClient.getOrganizationId());
        assertEquals(clientDTO.getLocationId(), createdClient.getLocationId());
    }

    @Test
    void testUpdateClient() throws Exception {
        // Arrange
        UpdateClientDTO clientDTO = new UpdateClientDTO(clientId, "Test Client - Updated Unique Title 123456789", locationId, null, false);
        String clientDTOJson = objectMapper.writeValueAsString(clientDTO);
        String invalidJWTToken = "Invalid";

        // Act (invalid security credentials)
        mockMvc.perform(put("/api/v1/clients/update")
                        .header("Authorization", "Bearer " + invalidJWTToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(clientDTOJson))
                        .andExpect(status().is(403));

        // Assert
        Optional<Client> invalidUpdatedClientOptional = clientRepository.findByName(clientDTO.getName());
        if (invalidUpdatedClientOptional.isPresent()) {
            fail("Failed to prevent update on invalid JWT token.");
        }

        // Act
        mockMvc.perform(put("/api/v1/clients/update")
                        .header("Authorization", "Bearer " + jwtToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(clientDTOJson))
                        .andExpect(status().isOk());

        // Assert
        Optional<Client> updatedClientOptional = clientRepository.findByName(clientDTO.getName());
        if (updatedClientOptional.isEmpty()) {
            fail("Updated client has not been found");
        }
        Client updatedClient = updatedClientOptional.get();
        assertNotNull(updatedClient);
        assertEquals(clientDTO.getName(), updatedClient.getName());
        assertEquals(clientDTO.getLocationId(), updatedClient.getLocationId());
    }

    @Test
    void testDeleteClient() throws Exception {
        // Arrange
        String url = "http://localhost:8080/api/v1/clients/delete/" + clientId;
        String invalidJWTToken = "Invalid";

        // Act (invalid security credentials)
        mockMvc.perform(delete(url)
                        .header("Authorization", "Bearer " + invalidJWTToken))
                        .andExpect(status().is(403));

        // Assert
        Optional<Client> invalidUpdatedClientOptional = clientRepository.findById(clientId);
        if (invalidUpdatedClientOptional.isEmpty()) {
            fail("Failed to prevent deletion on invalid JWT Token.");
        }

        // Act
        mockMvc.perform(delete(url)
                .header("Authorization", "Bearer " + jwtToken))
                .andExpect(status().isOk());

        // Assert
        Optional<Client> updatedClientOptional = clientRepository.findById(clientId);
        if (updatedClientOptional.isPresent()) {
            fail("Client has not been deleted as expected.");
        }
    }

}
