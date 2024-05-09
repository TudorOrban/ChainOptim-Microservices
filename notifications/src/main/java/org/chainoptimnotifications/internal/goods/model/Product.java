package org.chainoptimnotifications.internal.goods.model;

import lombok.*;
import org.chainoptimnotifications.internal.goods.model.UnitOfMeasurement;

import java.time.LocalDateTime;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Product {

    private Integer id;
    private String name;
    private String description;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer organizationId;
    private UnitOfMeasurement unit;

}
