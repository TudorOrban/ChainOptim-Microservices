package org.chainoptimnotifications.user.service;

import org.chainoptimnotifications.user.model.UserDetailsImpl;

public interface AuthenticationService {

    UserDetailsImpl loadUserByUsername(String username);
}
