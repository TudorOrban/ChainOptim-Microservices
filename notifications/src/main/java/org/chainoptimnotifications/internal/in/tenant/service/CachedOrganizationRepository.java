package org.chainoptimnotifications.internal.in.tenant.service;

import org.chainoptimnotifications.internal.in.tenant.model.Organization;

public interface CachedOrganizationRepository {

    Organization getOrganizationWithUsersAndCustomRoles(Integer id);
}
