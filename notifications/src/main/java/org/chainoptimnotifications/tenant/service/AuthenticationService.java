package org.chainoptimnotifications.tenant.service;

import org.chainoptimnotifications.tenant.model.UserDetailsImpl;

public interface AuthenticationService {

    UserDetailsImpl loadUserByUsername(String username);
}
