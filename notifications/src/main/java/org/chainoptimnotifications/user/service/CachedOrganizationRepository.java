package org.chainoptimnotifications.user.service;

import org.chainoptimnotifications.user.model.Organization;

public interface CachedOrganizationRepository {

    Organization getOrganizationWithUsersAndCustomRoles(Integer id);
}
