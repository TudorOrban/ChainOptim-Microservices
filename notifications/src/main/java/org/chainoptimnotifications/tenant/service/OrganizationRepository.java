package org.chainoptimnotifications.tenant.service;

import org.chainoptimnotifications.tenant.model.Organization;

public interface OrganizationRepository {

    Organization getOrganizationWithUsersAndCustomRoles(Integer id);
}
