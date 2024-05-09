package org.chainoptimdemand.core.performance.controller;

import org.chainoptimdemand.internal.in.security.service.SecurityService;
import org.chainoptimdemand.core.performance.dto.CreateClientPerformanceDTO;
import org.chainoptimdemand.core.performance.dto.UpdateClientPerformanceDTO;
import org.chainoptimdemand.core.performance.model.ClientPerformance;
import org.chainoptimdemand.core.performance.service.ClientPerformancePersistenceService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/client-performance")
public class ClientPerformanceController {

    private final ClientPerformancePersistenceService clientPerformancePersistenceService;
    private final SecurityService securityService;

    @Autowired
    public ClientPerformanceController(
            ClientPerformancePersistenceService clientPerformancePersistenceService,
            SecurityService securityService
    ) {
        this.clientPerformancePersistenceService = clientPerformancePersistenceService;
        this.securityService = securityService;
    }

    // Fetch
    @PreAuthorize("@securityService.canAccessEntity(#clientId, \"Client\", \"Read\")")
    @GetMapping("/client/{clientId}")
    public ResponseEntity<ClientPerformance> getClientPerformance(@PathVariable Integer clientId) {
        ClientPerformance clientPerformance = clientPerformancePersistenceService.getClientPerformance(clientId);
        return ResponseEntity.ok(clientPerformance);
    }

    @PreAuthorize("@securityService.canAccessEntity(#clientId, \"Client\", \"Read\")")
    @GetMapping("/client/{clientId}/refresh")
    public ResponseEntity<ClientPerformance> evaluateClientPerformance(@PathVariable Integer clientId) {
        ClientPerformance clientPerformance = clientPerformancePersistenceService.refreshClientPerformance(clientId);
        return ResponseEntity.ok(clientPerformance);
    }

    // Create
    @PreAuthorize("@securityService.canAccessEntity(#performanceDTO.getClientId(), \"Client\", \"Create\")")
    @PostMapping("/create")
    public ResponseEntity<ClientPerformance> createClientPerformance(@RequestBody CreateClientPerformanceDTO performanceDTO) {
        ClientPerformance createdClientPerformance = clientPerformancePersistenceService.createClientPerformance(performanceDTO);
        return ResponseEntity.ok(createdClientPerformance);
    }

    // Update
    @PreAuthorize("@securityService.canAccessEntity(#performanceDTO.getClientId(), \"Client\", \"Update\")")
    @PutMapping("/update")
    public ResponseEntity<ClientPerformance> updateClientPerformance(@RequestBody UpdateClientPerformanceDTO performanceDTO) {
        ClientPerformance updatedClientPerformance = clientPerformancePersistenceService.updateClientPerformance(performanceDTO);
        return ResponseEntity.ok(updatedClientPerformance);
    }

    // Delete
    @PreAuthorize("@securityService.canAccessEntity(#id, \"Client\", \"Delete\")")
    @DeleteMapping("/delete/{id}")
    public ResponseEntity<Void> deleteClientPerformance(@PathVariable Integer id) {
        clientPerformancePersistenceService.deleteClientPerformance(id);
        return ResponseEntity.noContent().build();
    }

}
