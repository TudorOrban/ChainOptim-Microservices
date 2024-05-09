package org.chainoptimdemand.internal.in.goods.model;

import lombok.*;

import java.time.LocalDateTime;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UnitOfMeasurement {

    private Integer id;
    private String name;
    private LocalDateTime createdAt;
    private String unitType;
    private Integer organizationId;
}
