package org.chainoptimnotifications.internal.demand.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.chainoptimnotifications.shared.enums.OrderStatus;
import org.chainoptimnotifications.internal.goods.model.Product;

import java.time.LocalDateTime;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class ClientOrder {

    private Integer id;
    private Integer clientId;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Product product;
    private Integer organizationId;
    private Float quantity;
    private Float deliveredQuantity;
    private LocalDateTime orderDate;
    private LocalDateTime estimatedDeliveryDate;
    private LocalDateTime deliveryDate;
    private OrderStatus status;
    private String companyId;

    public ClientOrder deepCopy() {
        return ClientOrder.builder()
//                .id(this.id)
                .createdAt(this.createdAt)
                .updatedAt(this.updatedAt)
                .organizationId(this.organizationId)
                .clientId(this.clientId)
                .product(this.product)
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
