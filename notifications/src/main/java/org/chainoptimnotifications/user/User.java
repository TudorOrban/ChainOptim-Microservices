package org.chainoptimnotifications.user;

import java.io.Serializable;
import java.time.LocalDateTime;

public class User {

    private String id;

    private String username;
    private String passwordHash;
    private String email;
    private java.time.LocalDateTime createdAt;
    private java.time.LocalDateTime updatedAt;
    private Organization organization;

    // Role
    public enum Role implements Serializable {
        ADMIN,
        MEMBER,
        NONE
    }

    private CustomRole customRole;
    private Boolean isProfilePublic;
    private String verificationToken;
    private LocalDateTime verificationTokenExpirationDate;
    private Boolean enabled;
    private Boolean isFirstConfirmationEmail;


}