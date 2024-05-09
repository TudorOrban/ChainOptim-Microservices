package org.chainoptimsupply.internal.tenant.service;

import org.chainoptimsupply.internal.tenant.model.UserDetailsImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.stereotype.Service;


@Service
public class UserDetailsServiceImpl implements UserDetailsService {

    private final AuthenticationService authenticationService;

    @Autowired
    public UserDetailsServiceImpl(AuthenticationService authenticationService) {
        this.authenticationService = authenticationService;
    }

    @Override
    public UserDetailsImpl loadUserByUsername(String username) {
        return authenticationService.loadUserByUsername(username);
    }


}
