package org.chainoptim.internalcommunication.controller;

import org.chainoptim.config.security.SecurityService;
import org.chainoptim.core.organization.model.Organization;
import org.chainoptim.core.organization.service.OrganizationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/internal/organizations")
public class OrganizationInternalController {

    private final OrganizationService organizationService;
    private final SecurityService securityService;

    @Autowired
    public OrganizationInternalController(OrganizationService organizationService,
                                          SecurityService securityService) {
        this.organizationService = organizationService;
        this.securityService = securityService;
    }

    @GetMapping("/{id}")
    public ResponseEntity<Organization> getOrganizationByIdWithUsersAndCustomRoles(@PathVariable Integer id) {
        Organization organization = organizationService.getOrganizationWithUsersAndCustomRoles(id);
        if (organization != null) {
            return ResponseEntity.ok(organization);
        } else {
            return ResponseEntity.notFound().build();
        }
    }
}
