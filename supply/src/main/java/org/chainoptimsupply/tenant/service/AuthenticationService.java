package org.chainoptimsupply.tenant.service;


import org.chainoptimsupply.tenant.model.UserDetailsImpl;

public interface AuthenticationService {

    UserDetailsImpl loadUserByUsername(String username);
}
