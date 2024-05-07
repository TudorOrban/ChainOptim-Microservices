package org.chainoptim.internalcommunication.controller;

import org.chainoptim.config.security.SecurityService;
import org.chainoptim.core.organization.model.Organization;
import org.chainoptim.core.organization.repository.OrganizationRepository;
import org.chainoptim.core.organization.service.OrganizationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Optional;

@RestController
@RequestMapping("/api/v1/internal/organizations")
public class OrganizationInternalController {

    private final OrganizationService organizationService;
    private final OrganizationRepository organizationRepository;
    private final SecurityService securityService;

    @Autowired
    public OrganizationInternalController(OrganizationService organizationService,
                                          OrganizationRepository organizationRepository,
                                          SecurityService securityService) {
        this.organizationService = organizationService;
        this.organizationRepository = organizationRepository;
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

    @GetMapping("/{id}/subscription-plan-tier")
    public ResponseEntity<Organization.SubscriptionPlanTier> getSubscriptionPlanTierById(@PathVariable Integer id) {
        Optional<Organization.SubscriptionPlanTier> subscriptionPlanTier = organizationRepository.getSubscriptionPlanTierById(id);
        return subscriptionPlanTier.map(ResponseEntity::ok).orElseGet(() -> ResponseEntity.notFound().build());
    }
}
