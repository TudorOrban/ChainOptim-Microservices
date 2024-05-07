package org.chainoptimsupply.tenant;

import org.chainoptimnotifications.user.model.Organization;

public interface OrganizationRepository {

    Organization getOrganizationWithUsersAndCustomRoles(Integer id);
}
