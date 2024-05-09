package org.chainoptimdemand.internal.in.tenant.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serial;
import java.io.Serializable;
import java.time.LocalDateTime;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class User implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private String id;

    private String username;
    private String passwordHash;
    private String email;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Organization organization;

    // Role
    public enum Role implements Serializable {
        ADMIN,
        MEMBER,
        NONE
    }

    private Role role;
    private CustomRole customRole;
    private Boolean isProfilePublic;
    private String verificationToken;
    private LocalDateTime verificationTokenExpirationDate;
    private Boolean enabled;
    private Boolean isFirstConfirmationEmail;


}