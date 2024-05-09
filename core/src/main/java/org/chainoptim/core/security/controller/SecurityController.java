package org.chainoptim.core.security.controller;

import org.chainoptim.core.security.dto.SecurityDTO;
import org.chainoptim.core.security.service.SecurityService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Optional;

@RestController
@RequestMapping("/api/v1/internal/security")
public class SecurityController {

    private final SecurityService securityService;

    @Autowired
    public SecurityController(SecurityService securityService) {
        this.securityService = securityService;
    }

    @PostMapping("/can-access")
    public ResponseEntity<Boolean> canAccess(@RequestBody SecurityDTO securityDTO) {
        boolean canAccess = securityService.canAccessOrganizationEntity(Optional.of(securityDTO.getOrganizationId()), securityDTO.getEntityType(), securityDTO.getOperationType());
        return ResponseEntity.ok(canAccess);
    }
}
