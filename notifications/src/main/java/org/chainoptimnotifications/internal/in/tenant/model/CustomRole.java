package org.chainoptimnotifications.internal.in.tenant.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serial;
import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.Set;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class CustomRole implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private Integer id;
    private String name;
    private Integer organizationId;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Set<User> users;
    private Permissions permissions;
}
