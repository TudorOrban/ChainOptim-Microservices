package org.chainoptim.core.security.service;

import org.chainoptim.core.user.model.UserDetailsImpl;

public interface CustomRoleSecurityService {

    boolean canUserAccessOrganizationEntity(Integer organizationId, UserDetailsImpl userDetails, String entityType, String operationType);
}
