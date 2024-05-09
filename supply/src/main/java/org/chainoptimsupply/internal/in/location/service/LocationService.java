package org.chainoptimsupply.internal.in.location.service;

import org.chainoptimsupply.internal.in.location.dto.CreateLocationDTO;
import org.chainoptimsupply.internal.in.location.dto.Location;

public interface LocationService {

    Location createLocation(CreateLocationDTO locationDTO);
}
