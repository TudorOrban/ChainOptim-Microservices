package org.chainoptimdemand.internal.in.location.service;

import org.chainoptimdemand.internal.in.location.dto.CreateLocationDTO;
import org.chainoptimdemand.internal.in.location.dto.Location;

public interface LocationService {

    Location createLocation(CreateLocationDTO locationDTO);
}
