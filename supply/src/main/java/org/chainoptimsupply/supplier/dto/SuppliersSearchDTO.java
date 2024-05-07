package org.chainoptimsupply.supplier.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.chainoptimsupply.shared.dto.Location;

import java.time.LocalDateTime;

@Getter
@Setter
@NoArgsConstructor
public class SuppliersSearchDTO {
    private Integer id;
    private String name;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer locationId;
}
