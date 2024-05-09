package org.chainoptimnotifications.internal.in.tenant.service;

import org.chainoptimnotifications.internal.in.tenant.model.Organization;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;

@Service
public class CachedOrganizationRepositoryImpl implements CachedOrganizationRepository {

    private final OrganizationRepository organizationRepository;

    @Autowired
    public CachedOrganizationRepositoryImpl(OrganizationRepository organizationRepository) {
        this.organizationRepository = organizationRepository;
    }

    @Cacheable(value = "organizationCache", key="#id")
    public Organization getOrganizationWithUsersAndCustomRoles(Integer id) {
        System.out.println("Fetching organization with id: " + id);
        return organizationRepository.getOrganizationWithUsersAndCustomRoles(id);
    }


}
