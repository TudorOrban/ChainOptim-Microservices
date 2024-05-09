package org.chainoptimdemand.client.service;

import org.chainoptimdemand.core.client.service.ClientServiceImpl;
import org.chainoptimdemand.exception.ResourceNotFoundException;
import org.chainoptimdemand.internal.in.tenant.service.SubscriptionPlanLimiterService;
import org.chainoptimdemand.internal.in.location.dto.CreateLocationDTO;
import org.chainoptimdemand.shared.sanitization.EntitySanitizerService;
import org.chainoptimdemand.core.client.dto.CreateClientDTO;
import org.chainoptimdemand.core.client.dto.ClientDTOMapper;
import org.chainoptimdemand.core.client.dto.UpdateClientDTO;
import org.chainoptimdemand.core.client.model.Client;
import org.chainoptimdemand.core.client.repository.ClientRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ClientServiceTest {

    @Mock
    private ClientRepository clientRepository;
    @Mock
    private SubscriptionPlanLimiterService planLimiterService;
    @Mock
    private EntitySanitizerService entitySanitizerService;

    @InjectMocks
    private ClientServiceImpl clientService;

    @Test
    void testCreateClient() {
        // Arrange
        CreateClientDTO clientDTO = new CreateClientDTO("Test Client", 1, 1, new CreateLocationDTO(), false);
        Client expectedClient = ClientDTOMapper.convertCreateClientDTOToClient(clientDTO);

        when(clientRepository.save(any(Client.class))).thenReturn(expectedClient);
        when(planLimiterService.isLimitReached(any(), any(), any())).thenReturn(false);
        when(entitySanitizerService.sanitizeCreateClientDTO(any(CreateClientDTO.class))).thenReturn(clientDTO);

        // Act
        Client createdClient = clientService.createClient(clientDTO);

        // Assert
        assertNotNull(createdClient);
        assertEquals(expectedClient.getName(), createdClient.getName());
        assertEquals(expectedClient.getOrganizationId(), createdClient.getOrganizationId());
        assertEquals(expectedClient.getLocationId(), createdClient.getLocationId());

        verify(clientRepository, times(1)).save(any(Client.class));
    }

    @Test
    void testUpdateClient_ExistingClient() {
        // Arrange
        UpdateClientDTO clientDTO = new UpdateClientDTO(1, "Test Client", 1, new CreateLocationDTO(), false);
        Client existingClient = new Client();
        existingClient.setId(1);

        when(clientRepository.findById(1)).thenReturn(Optional.of(existingClient));
        when(clientRepository.save(any(Client.class))).thenReturn(existingClient);
        when(entitySanitizerService.sanitizeUpdateClientDTO(any(UpdateClientDTO.class))).thenReturn(clientDTO);

        // Act
        Client updatedClient = clientService.updateClient(clientDTO);

        // Assert
        assertNotNull(updatedClient);
        assertEquals(existingClient.getName(), updatedClient.getName());
        assertEquals(existingClient.getOrganizationId(), updatedClient.getOrganizationId());
        assertEquals(existingClient.getLocationId(), updatedClient.getLocationId());

        verify(clientRepository, times(1)).findById(1);
    }

    @Test
    void testUpdateClient_NonExistingClient() {
        // Arrange
        UpdateClientDTO clientDTO = new UpdateClientDTO(1, "Test Client", 1, new CreateLocationDTO(), false);
        Client existingClient = new Client();
        existingClient.setId(1);

        when(clientRepository.findById(1)).thenReturn(Optional.empty());
        when(entitySanitizerService.sanitizeUpdateClientDTO(any(UpdateClientDTO.class))).thenReturn(clientDTO);

        // Act and assert
        assertThrows(ResourceNotFoundException.class, () -> clientService.updateClient(clientDTO));

        verify(clientRepository, times(1)).findById(1);
        verify(clientRepository, never()).save(any(Client.class));
    }

    @Test
    void testDeleteClient() {
        // Arrange
        doNothing().when(clientRepository).delete(any(Client.class));

        // Act
        clientService.deleteClient(1);

        // Assert
        verify(clientRepository, times(1)).delete(any(Client.class));
    }
}
