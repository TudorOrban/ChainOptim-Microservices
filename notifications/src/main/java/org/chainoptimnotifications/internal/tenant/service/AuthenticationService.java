package org.chainoptimnotifications.internal.tenant.service;

import org.chainoptimnotifications.internal.tenant.model.UserDetailsImpl;

public interface AuthenticationService {

    UserDetailsImpl loadUserByUsername(String username);
}
