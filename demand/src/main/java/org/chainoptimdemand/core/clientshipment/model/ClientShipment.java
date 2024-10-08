package org.chainoptimdemand.core.clientshipment.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Entity
@Table(name = "client_shipments")
public class ClientShipment {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    @Column(name = "client_order_id", nullable = false)
    private Integer clientOrderId;

    @Column(name = "quantity")
    private Float quantity;

    @Column(name = "shipment_starting_date")
    private LocalDateTime shipmentStartingDate;

    @Column(name = "estimated_arrival_date")
    private LocalDateTime estimatedArrivalDate;

    @Column(name = "arrival_date")
    private LocalDateTime arrivalDate;

    @Column(name = "transporter_type")
    private String transporterType;

    @Column(name = "status")
    private String status;

    @Column(name = "source_location_id")
    private Integer sourceLocationId;

    @Column(name = "destination_location_id")
    private Integer destinationLocationId;

    @Column(name = "current_location_latitude")
    private Float currentLocationLatitude;

    @Column(name = "current_location_longitude")
    private Float currentLocationLongitude;

}
