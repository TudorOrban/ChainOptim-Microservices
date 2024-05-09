package org.chainoptimnotifications.internal.in.goods.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class UnitOfMeasurement {

    private Integer id;
    private String name;
    private LocalDateTime createdAt;
    private String unitType;
    private Integer organizationId;
}
