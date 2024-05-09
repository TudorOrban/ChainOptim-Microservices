package org.chainoptimnotifications.internal.in.tenant.service;

import org.chainoptimnotifications.internal.in.tenant.model.UserDetailsImpl;

public interface AuthenticationService {

    UserDetailsImpl loadUserByUsername(String username);
}
