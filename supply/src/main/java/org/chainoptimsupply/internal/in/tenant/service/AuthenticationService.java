package org.chainoptimsupply.internal.in.tenant.service;


import org.chainoptimsupply.internal.in.tenant.model.UserDetailsImpl;

public interface AuthenticationService {

    UserDetailsImpl loadUserByUsername(String username);
}
