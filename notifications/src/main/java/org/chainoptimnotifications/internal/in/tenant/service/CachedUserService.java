package org.chainoptimnotifications.internal.in.tenant.service;

import org.chainoptimnotifications.internal.in.tenant.model.UserDetailsImpl;

public interface CachedUserService {

    UserDetailsImpl cachedLoadUserByUsername(String username);
//    User assignBasicRoleToUser(String userId, User.Role role);
//    User assignCustomRoleToUser(String userId, Integer roleId);
}
