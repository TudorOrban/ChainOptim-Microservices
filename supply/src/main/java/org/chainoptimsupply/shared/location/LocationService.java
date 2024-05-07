package org.chainoptimsupply.shared.location;

import org.chainoptimsupply.shared.dto.CreateLocationDTO;
import org.chainoptimsupply.shared.dto.Location;

public interface LocationService {

    Location createLocation(CreateLocationDTO locationDTO);
}
