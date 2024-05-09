package org.chainoptimnotifications.internal.tenant.service;

import org.chainoptimnotifications.internal.tenant.model.Organization;

public interface CachedOrganizationRepository {

    Organization getOrganizationWithUsersAndCustomRoles(Integer id);
}
