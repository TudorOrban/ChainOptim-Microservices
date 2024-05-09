package org.chainoptimnotifications.internal.tenant.service;

import org.chainoptimnotifications.internal.tenant.model.Organization;

public interface OrganizationRepository {

    Organization getOrganizationWithUsersAndCustomRoles(Integer id);
}
