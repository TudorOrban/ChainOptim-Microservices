package org.chainoptimstorage.core.warehouse.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.chainoptimstorage.internal.in.location.dto.Location;

import java.time.LocalDateTime;

@Getter
@Setter
@NoArgsConstructor
public class WarehousesSearchDTO {
    private Integer id;
    private String name;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer locationId;
}
