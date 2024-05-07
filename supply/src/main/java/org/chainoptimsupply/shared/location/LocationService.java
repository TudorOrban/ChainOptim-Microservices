package org.chainoptimsupply.shared.location;

import org.chainoptimsupply.shared.dto.CreateLocationDTO;
import org.chainoptimsupply.shared.dto.Location;
import org.chainoptimsupply.shared.dto.UpdateLocationDTO;

import java.util.List;

public interface LocationService {

    List<Location> getLocationsByOrganizationId(Integer organizationId);
    Location createLocation(CreateLocationDTO locationDTO);
    Location updateLocation(UpdateLocationDTO locationDTO);
    void deleteLocation(Integer locationId);
}
