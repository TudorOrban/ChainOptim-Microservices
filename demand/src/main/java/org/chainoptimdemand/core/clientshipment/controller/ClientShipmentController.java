package org.chainoptimdemand.core.clientshipment.controller;

import org.chainoptimdemand.internal.in.security.service.SecurityService;
import org.chainoptimdemand.shared.PaginatedResults;
import org.chainoptimdemand.core.client.dto.CreateClientShipmentDTO;
import org.chainoptimdemand.core.client.dto.UpdateClientShipmentDTO;
import org.chainoptimdemand.core.clientshipment.model.ClientShipment;
import org.chainoptimdemand.core.clientshipment.service.ClientShipmentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/v1/client-shipments")
public class ClientShipmentController {

    private final ClientShipmentService clientShipmentService;
    private final SecurityService securityService;

    @Autowired
    public ClientShipmentController(ClientShipmentService clientShipmentService,
                                      SecurityService securityService) {
        this.clientShipmentService = clientShipmentService;
        this.securityService = securityService;
    }

    @PreAuthorize("@securityService.canAccessEntity(#clientOrderId, \"ClientOrder\", \"Read\")")
    @GetMapping("/order/{clientOrderId}")
    public ResponseEntity<List<ClientShipment>> getClientShipmentsByClientOrderId(@PathVariable("clientOrderId") Integer clientOrderId) {
        List<ClientShipment> clientShipments = clientShipmentService.getClientShipmentsByClientOrderId(clientOrderId);
        return ResponseEntity.ok(clientShipments);
    }

    @PreAuthorize("@securityService.canAccessEntity(#clientOrderId, \"ClientOrder\", \"Read\")")
    @GetMapping("/order/advanced/{clientOrderId}")
    public ResponseEntity<PaginatedResults<ClientShipment>> getClientShipmentsByClientOrderIdAdvanced(
            @PathVariable Integer clientOrderId,
            @RequestParam(name = "searchQuery", required = false, defaultValue = "") String searchQuery,
            @RequestParam(name = "sortBy", required = false, defaultValue = "createdAt") String sortBy,
            @RequestParam(name = "ascending", required = false, defaultValue = "true") boolean ascending,
            @RequestParam(name = "page", required = false, defaultValue = "1") int page,
            @RequestParam(name = "itemsPerPage", required = false, defaultValue = "30") int itemsPerPage
    ) {
        PaginatedResults<ClientShipment> clientShipments = clientShipmentService.getClientShipmentsByClientOrderIdAdvanced(clientOrderId, searchQuery, sortBy, ascending, page, itemsPerPage);
        return ResponseEntity.ok(clientShipments);
    }

    // Create
    @PreAuthorize("@securityService.canAccessOrganizationEntity(#shipmentDTO.getOrganizationId(), \"Client\", \"Create\")")
    @PostMapping("/create")
    public ResponseEntity<ClientShipment> createClientShipment(@RequestBody CreateClientShipmentDTO shipmentDTO) {
        ClientShipment clientShipment = clientShipmentService.createClientShipment(shipmentDTO);
        return ResponseEntity.ok(clientShipment);
    }

    @PreAuthorize("@securityService.canAccessOrganizationEntity(#shipmentDTOs.getFirst().getOrganizationId(), \"Client\", \"Create\")")
    @PostMapping("/create/bulk")
    public ResponseEntity<List<ClientShipment>> createClientShipmentsInBulk(@RequestBody List<CreateClientShipmentDTO> shipmentDTOs) {
        List<ClientShipment> clientShipments = clientShipmentService.createClientShipmentsInBulk(shipmentDTOs);
        return ResponseEntity.ok(clientShipments);
    }

    @PreAuthorize("@securityService.canAccessOrganizationEntity(#shipmentDTOs.getFirst().getOrganizationId(), \"Client\", \"Update\")")
    @PutMapping("/update/bulk")
    public ResponseEntity<List<ClientShipment>> updateClientShipmentsInBulk(@RequestBody List<UpdateClientShipmentDTO> shipmentDTOs) {
        List<ClientShipment> clientShipments = clientShipmentService.updateClientShipmentsInBulk(shipmentDTOs);
        return ResponseEntity.ok(clientShipments);
    }

    // TODO: Secure endpoint
//    @PreAuthorize("@securityService.canAccessOrganizationEntity(#shipmentIds.getFirst(), \"ClientShipment\", \"Delete\")")
    @DeleteMapping("/delete/bulk")
    public ResponseEntity<List<Integer>> deleteClientShipmentsInBulk(@RequestBody List<Integer> shipmentIds) {
        clientShipmentService.deleteClientShipmentsInBulk(shipmentIds);

        return ResponseEntity.ok().build();
    }
}
