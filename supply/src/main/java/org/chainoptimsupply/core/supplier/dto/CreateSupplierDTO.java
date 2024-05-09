package org.chainoptimsupply.core.supplier.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.chainoptimsupply.internal.location.dto.CreateLocationDTO;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class CreateSupplierDTO {

    private String name;
    private Integer organizationId;
    private Integer locationId;
    private CreateLocationDTO location;
    private boolean createLocation;
}
