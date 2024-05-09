package org.chainoptimdemand.internal.out.clientorder.controller;

import org.chainoptimdemand.core.clientorder.model.ClientOrder;
import org.chainoptimdemand.core.clientorder.service.ClientOrderService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/v1/internal/client-order")
public class ClientOrderInternalController {

    private final ClientOrderService clientOrderService;

    @Autowired
    public ClientOrderInternalController(ClientOrderService clientOrderService) {
        this.clientOrderService = clientOrderService;
    }

    @GetMapping("/organization/{organizationId}")
    public List<ClientOrder> getClientOrdersByOrganizationId(@PathVariable Integer organizationId) {
        return clientOrderService.getClientOrdersByOrganizationId(organizationId);
    }

    @GetMapping("/organization/{organizationId}/count")
    public long countByOrganizationId(@PathVariable Integer organizationId) {
        return clientOrderService.countByOrganizationId(organizationId);
    }

    @GetMapping("/{clientOrderId}/organization-id")
    public Integer getOrganizationIdById(@PathVariable Long clientOrderId) {
        return clientOrderService.getOrganizationIdById(clientOrderId);
    }
}
