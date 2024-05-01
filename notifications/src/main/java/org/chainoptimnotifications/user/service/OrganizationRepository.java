package org.chainoptimnotifications.user.service;

import org.chainoptimnotifications.user.model.Organization;

public interface OrganizationRepository {

    Organization getOrganizationWithUsersAndCustomRoles(Integer id);
}
