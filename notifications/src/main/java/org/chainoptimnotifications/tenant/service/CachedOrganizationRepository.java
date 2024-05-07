package org.chainoptimnotifications.tenant.service;

import org.chainoptimnotifications.tenant.model.Organization;

public interface CachedOrganizationRepository {

    Organization getOrganizationWithUsersAndCustomRoles(Integer id);
}
