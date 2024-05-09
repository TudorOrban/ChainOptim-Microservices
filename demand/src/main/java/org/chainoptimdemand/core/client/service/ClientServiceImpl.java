package org.chainoptimdemand.core.client.service;

import org.chainoptimdemand.core.client.dto.CreateClientDTO;
import org.chainoptimdemand.core.client.dto.ClientDTOMapper;
import org.chainoptimdemand.core.client.dto.ClientsSearchDTO;
import org.chainoptimdemand.core.client.dto.UpdateClientDTO;
import org.chainoptimdemand.core.client.model.Client;
import org.chainoptimdemand.core.client.repository.ClientRepository;
import org.chainoptimdemand.exception.PlanLimitReachedException;
import org.chainoptimdemand.exception.ResourceNotFoundException;
import org.chainoptimdemand.exception.ValidationException;
import org.chainoptimdemand.internal.in.tenant.service.SubscriptionPlanLimiterService;
import org.chainoptimdemand.shared.enums.Feature;
import org.chainoptimdemand.shared.PaginatedResults;
import org.chainoptimdemand.internal.in.location.dto.Location;
import org.chainoptimdemand.internal.in.location.service.LocationService;
import org.chainoptimdemand.shared.sanitization.EntitySanitizerService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ClientServiceImpl implements ClientService {

    private final ClientRepository clientRepository;
    private final LocationService locationService;
    private final SubscriptionPlanLimiterService planLimiterService;
    private final EntitySanitizerService entitySanitizerService;

    @Autowired
    public ClientServiceImpl(ClientRepository clientRepository,
                               LocationService locationService,
                               SubscriptionPlanLimiterService planLimiterService,
                               EntitySanitizerService entitySanitizerService) {
        this.clientRepository = clientRepository;
        this.locationService = locationService;
        this.planLimiterService = planLimiterService;
        this.entitySanitizerService = entitySanitizerService;
    }

    public List<Client> getAllClients() {
        return clientRepository.findAll();
    }

    public Client getClientById(Integer clientId) {
        return clientRepository.findById(clientId)
                .orElseThrow(() -> new ResourceNotFoundException("Client with ID: " + clientId + " not found."));
    }

    public List<Client> getClientsByOrganizationId(Integer organizationId) {
        return clientRepository.findByOrganizationId(organizationId);
    }

    public PaginatedResults<ClientsSearchDTO> getClientsByOrganizationIdAdvanced(Integer organizationId, String searchQuery, String sortBy, boolean ascending, int page, int itemsPerPage) {
        PaginatedResults<Client> paginatedResults = clientRepository.findByOrganizationIdAdvanced(organizationId, searchQuery, sortBy, ascending, page, itemsPerPage);
        return new PaginatedResults<>(
            paginatedResults.results.stream()
            .map(ClientDTOMapper::convertToClientsSearchDTO)
            .toList(),
            paginatedResults.totalCount
        );
    }

    public Integer getOrganizationIdById(Long clientId) {
        return clientRepository.findOrganizationIdById(clientId)
                .orElseThrow(() -> new ResourceNotFoundException("Client with ID: " + clientId + " not found."));
    }

    public long countByOrganizationId(Integer organizationId) {
        return clientRepository.countByOrganizationId(organizationId);
    }

    // Create
    public Client createClient(CreateClientDTO clientDTO) {
        // Check if plan limit is reached
        if (planLimiterService.isLimitReached(clientDTO.getOrganizationId(), Feature.SUPPLIER, 1)) {
            throw new PlanLimitReachedException("You have reached the limit of allowed clients for the current Subscription Plan.");
        }

        // Sanitize input
        CreateClientDTO sanitizedClientDTO = entitySanitizerService.sanitizeCreateClientDTO(clientDTO);

        // Create location if requested
        if (sanitizedClientDTO.isCreateLocation() && sanitizedClientDTO.getLocation() != null) {
            Location location = locationService.createLocation(sanitizedClientDTO.getLocation());
            Client client = ClientDTOMapper.convertCreateClientDTOToClient(sanitizedClientDTO);
            client.setLocationId(location.getId());
            return clientRepository.save(client);
        } else {
            return clientRepository.save(ClientDTOMapper.convertCreateClientDTOToClient(sanitizedClientDTO));
        }
    }

    public Client updateClient(UpdateClientDTO clientDTO) {
        UpdateClientDTO sanitizedClientDTO = entitySanitizerService.sanitizeUpdateClientDTO(clientDTO);

        Client client = clientRepository.findById(sanitizedClientDTO.getId())
                .orElseThrow(() -> new ResourceNotFoundException("Client with ID: " + sanitizedClientDTO.getId() + " not found."));

        client.setName(sanitizedClientDTO.getName());

        // Create new client or use existing or throw if not provided
        Location location;
        if (sanitizedClientDTO.isCreateLocation() && sanitizedClientDTO.getLocation() != null) {
            location = locationService.createLocation(sanitizedClientDTO.getLocation());
        } else if (sanitizedClientDTO.getLocationId() != null) {
            location = new Location();
            location.setId(sanitizedClientDTO.getLocationId());
        } else {
            throw new ValidationException("Location is required.");
        }
        client.setLocationId(location.getId());

        clientRepository.save(client);
        return client;
    }

    public void deleteClient(Integer clientId) {
        Client client = new Client();
        client.setId(clientId);
        clientRepository.delete(client);
    }
}
