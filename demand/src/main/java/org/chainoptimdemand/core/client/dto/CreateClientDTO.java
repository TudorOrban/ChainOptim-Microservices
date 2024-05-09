package org.chainoptimdemand.core.client.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.chainoptimdemand.internal.in.location.dto.CreateLocationDTO;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class CreateClientDTO {

    private String name;
    private Integer organizationId;
    private Integer locationId;
    private CreateLocationDTO location;
    private boolean createLocation;
}
