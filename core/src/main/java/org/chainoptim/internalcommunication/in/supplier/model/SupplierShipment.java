package org.chainoptim.internalcommunication.in.supplier.model;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.chainoptim.shared.commonfeatures.location.model.Location;

import java.time.LocalDateTime;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class SupplierShipment {

    private Integer id;
    private Integer supplierOrderId;
    private Float quantity;
    private LocalDateTime shipmentStartingDate;
    private LocalDateTime estimatedArrivalDate;
    private LocalDateTime arrivalDate;
    private String transporterType;
    private String status;
    private Location sourceLocation;
    private Location destinationLocation;
    private Float currentLocationLatitude;
    private Float currentLocationLongitude;

}
