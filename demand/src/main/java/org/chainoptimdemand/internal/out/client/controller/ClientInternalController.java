package org.chainoptimdemand.internal.out.client.controller;

import org.chainoptimdemand.core.client.model.Client;
import org.chainoptimdemand.core.client.service.ClientService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/v1/internal/clients")
public class ClientInternalController {

    private final ClientService clientService;

    @Autowired
    public ClientInternalController(ClientService clientService) {
        this.clientService = clientService;
    }

    @GetMapping("/organization/{organizationId}")
    public List<Client> getClientsByOrganizationId(@PathVariable Integer organizationId) {
        return clientService.getClientsByOrganizationId(organizationId);
    }

    @GetMapping("/organization/{organizationId}/count")
    public long countByOrganizationId(@PathVariable Integer organizationId) {
        return clientService.countByOrganizationId(organizationId);
    }

    @GetMapping("/{clientId}/organization-id")
    public Integer getOrganizationIdById(@PathVariable Long clientId) {
        return clientService.getOrganizationIdById(clientId);
    }
}
