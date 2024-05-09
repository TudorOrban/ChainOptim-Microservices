package org.chainoptim.core.security.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Optional;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class SecurityDTO {

    private Integer organizationId;
    private String entityType;
    private String operationType;
}
