package org.chainoptimstorage.internal.in.location.service;

import org.chainoptimstorage.internal.in.location.dto.CreateLocationDTO;
import org.chainoptimstorage.internal.in.location.dto.Location;

public interface LocationService {

    Location createLocation(CreateLocationDTO locationDTO);
}
