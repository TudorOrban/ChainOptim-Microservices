package org.chainoptim.internalcommunication.in.supplier.model;

import lombok.*;
import org.chainoptim.shared.commonfeatures.location.model.Location;
import java.time.LocalDateTime;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Supplier {

    private Integer id;
    private String name;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Location location;
    private Integer organizationId;
}
