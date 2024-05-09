package org.chainoptimnotifications.internal.in.tenant.service;

import org.chainoptimnotifications.internal.in.tenant.model.Organization;

public interface OrganizationRepository {

    Organization getOrganizationWithUsersAndCustomRoles(Integer id);
}
