package org.chainoptimnotifications.user.service;

import org.chainoptimnotifications.user.model.User;
import org.chainoptimnotifications.user.model.UserDetailsImpl;

public interface CachedUserService {

    UserDetailsImpl cachedLoadUserByUsername(String username);
//    User assignBasicRoleToUser(String userId, User.Role role);
//    User assignCustomRoleToUser(String userId, Integer roleId);
}
