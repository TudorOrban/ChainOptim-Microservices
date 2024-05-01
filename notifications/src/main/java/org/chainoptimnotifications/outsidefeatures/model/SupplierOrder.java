package org.chainoptimnotifications.outsidefeatures.model;

import jakarta.persistence.*;
import lombok.*;
import org.chainoptimnotifications.enums.OrderStatus;

import java.time.LocalDateTime;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SupplierOrder {

    private Integer id;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer organizationId;
    private Integer supplierId;
    private Component component;
    private Float quantity;
    private Float deliveredQuantity;
    private LocalDateTime orderDate;
    private LocalDateTime estimatedDeliveryDate;
    private LocalDateTime deliveryDate;
    private OrderStatus status;
    private String companyId;

    public SupplierOrder deepCopy() {
        return SupplierOrder.builder()
//                .id(this.id)
                .createdAt(this.createdAt)
                .updatedAt(this.updatedAt)
                .organizationId(this.organizationId)
                .supplierId(this.supplierId)
                .component(this.component)
                .quantity(this.quantity)
                .deliveredQuantity(this.deliveredQuantity)
                .orderDate(this.orderDate)
                .estimatedDeliveryDate(this.estimatedDeliveryDate)
                .deliveryDate(this.deliveryDate)
                .status(this.status)
                .companyId(this.companyId)
                .build();
    }
}
