package org.chainoptimdemand.internal.in.location.dto;

import lombok.*;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class Location {

    private Integer id;
    private String address;
    private String city;
    private String state;
    private String country;
    private String zipCode;
    private Double latitude;
    private Double longitude;
    private Integer organizationId;
}
