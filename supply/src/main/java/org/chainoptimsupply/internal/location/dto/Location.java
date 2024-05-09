package org.chainoptimsupply.internal.location.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Location {

    private Integer id;
    private String address;
    private String city;
    private String state;
    private String country;
    private Double latitude;
    private Double longitude;
    private String zipCode;
    private Integer organizationId;
}
