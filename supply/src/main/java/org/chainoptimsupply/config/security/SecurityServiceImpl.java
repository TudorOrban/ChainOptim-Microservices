package org.chainoptimsupply.config.security;

import org.springframework.stereotype.Service;

import java.util.Optional;


@Service("securityService")
public class SecurityServiceImpl implements SecurityService {

    public boolean canAccessEntity(Long entityId, String entityType, String operationType) {
        return true; // TODO: Call core backend to get authorization
    }

    public boolean canAccessOrganizationEntity(Optional<Integer> organizationId, String entityType, String operationType) {
        return true; // TODO: Call core backend to get authorization
    }

    public boolean canUserAccessOrganizationEntity(String userId, String operationType) {
        return true; // TODO: Call core backend to get authorization
    }
}
