package org.chainoptimstorage.internal.in.location.dto;



public class LocationDTOMapper {

    private LocationDTOMapper() {}

    public static Location convertCreateLocationDTOToLocation(CreateLocationDTO locationDTO) {
        Location location = new Location();
        location.setAddress(locationDTO.getAddress());
        location.setCity(locationDTO.getCity());
        location.setState(locationDTO.getState());
        location.setCountry(locationDTO.getCountry());
        location.setZipCode(locationDTO.getZipCode());
        location.setLatitude(locationDTO.getLatitude());
        location.setLongitude(locationDTO.getLongitude());
        location.setOrganizationId(locationDTO.getOrganizationId());

        return location;
    }

    public static Location updateLocationFromUpdateLocationDTO(Location location, UpdateLocationDTO locationDTO) {
        location.setAddress(locationDTO.getAddress());
        location.setCity(locationDTO.getCity());
        location.setState(locationDTO.getState());
        location.setCountry(locationDTO.getCountry());
        location.setZipCode(locationDTO.getZipCode());
        location.setLatitude(locationDTO.getLatitude());
        location.setLongitude(locationDTO.getLongitude());

        return location;
    }

}
