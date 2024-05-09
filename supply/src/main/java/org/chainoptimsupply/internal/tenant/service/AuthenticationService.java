package org.chainoptimsupply.internal.tenant.service;


import org.chainoptimsupply.internal.tenant.model.UserDetailsImpl;

public interface AuthenticationService {

    UserDetailsImpl loadUserByUsername(String username);
}
