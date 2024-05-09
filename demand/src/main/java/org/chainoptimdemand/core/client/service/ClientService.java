package org.chainoptimdemand.core.client.service;


import org.chainoptimdemand.core.client.dto.CreateClientDTO;
import org.chainoptimdemand.core.client.dto.ClientsSearchDTO;
import org.chainoptimdemand.core.client.dto.UpdateClientDTO;
import org.chainoptimdemand.core.client.model.Client;
import org.chainoptimdemand.shared.PaginatedResults;

import java.util.List;

public interface ClientService {
    // Fetch
    List<Client> getAllClients();
    Client getClientById(Integer id);
    List<Client> getClientsByOrganizationId(Integer organizationId);
    PaginatedResults<ClientsSearchDTO> getClientsByOrganizationIdAdvanced(Integer organizationId, String searchQuery, String sortBy, boolean ascending, int page, int itemsPerPage);
    Integer getOrganizationIdById(Long clientId);
    long countByOrganizationId(Integer organizationId);

    // Create
    Client createClient(CreateClientDTO clientDTO);

    // Update
    Client updateClient(UpdateClientDTO updateClientDTO);

    // Delete
    void deleteClient(Integer clientId);
}
