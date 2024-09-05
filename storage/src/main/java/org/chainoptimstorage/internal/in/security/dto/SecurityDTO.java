package org.chainoptimstorage.internal.in.security.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class SecurityDTO {

    private Integer organizationId;
    private String entityType;
    private String operationType;
}
