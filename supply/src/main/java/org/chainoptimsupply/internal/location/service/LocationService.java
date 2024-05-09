package org.chainoptimsupply.internal.location.service;

import org.chainoptimsupply.internal.location.dto.CreateLocationDTO;
import org.chainoptimsupply.internal.location.dto.Location;

public interface LocationService {

    Location createLocation(CreateLocationDTO locationDTO);
}
