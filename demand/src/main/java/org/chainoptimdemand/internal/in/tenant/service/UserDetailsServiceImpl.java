package org.chainoptimdemand.internal.in.tenant.service;

import org.chainoptimdemand.internal.in.tenant.model.UserDetailsImpl;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.stereotype.Service;


@Service
public class UserDetailsServiceImpl implements UserDetailsService {


    @Override
    public UserDetailsImpl loadUserByUsername(String username) {
        return new UserDetailsImpl();
    }
}
