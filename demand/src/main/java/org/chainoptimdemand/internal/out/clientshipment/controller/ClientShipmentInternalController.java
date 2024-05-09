package org.chainoptimdemand.internal.out.clientshipment.controller;

import org.chainoptimdemand.core.clientshipment.model.ClientShipment;
import org.chainoptimdemand.core.clientshipment.service.ClientShipmentService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/v1/internal/client-shipments")
public class ClientShipmentInternalController {

    private final ClientShipmentService clientShipmentService;

    @Autowired
    public ClientShipmentInternalController(ClientShipmentService clientShipmentService) {
        this.clientShipmentService = clientShipmentService;
    }

    @GetMapping("/client-orders/{orderIds}")
    public List<ClientShipment> getClientShipmentsByClientOrderIds(@PathVariable List<Integer> orderIds) {
        return clientShipmentService.getClientShipmentsByClientOrderIds(orderIds);
    }

    @GetMapping("/organization/{organizationId}/count")
    public long getClientShipmentsByOrganizationId(@PathVariable Integer organizationId) {
        return clientShipmentService.countByOrganizationId(organizationId);
    }
}
