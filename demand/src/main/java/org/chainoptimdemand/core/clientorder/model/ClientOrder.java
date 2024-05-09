package org.chainoptimdemand.core.clientorder.model;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Entity
@Table(name = "client_orders")
public class ClientOrder {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id", updatable = false, nullable = false)
    private Integer id;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    @Column(name = "organization_id", nullable = false)
    private Integer organizationId;

    @Column(name = "client_id", nullable = false)
    private Integer clientId;

    @Column(name = "component_id")
    private Integer componentId;

    @Column(name = "quantity")
    private Float quantity;

    @Column(name = "delivered_quantity")
    private Float deliveredQuantity;

    @Column(name = "order_date", nullable = true)
    private LocalDateTime orderDate;

    @Column(name = "estimated_delivery_date", nullable = true)
    private LocalDateTime estimatedDeliveryDate;

    @Column(name = "delivery_date", nullable = true)
    private LocalDateTime deliveryDate;

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false)
    private OrderStatus status;

    @Column(name = "company_id")
    private String companyId;

    public ClientOrder deepCopy() {
        return ClientOrder.builder()
//                .id(this.id)
                .createdAt(this.createdAt)
                .updatedAt(this.updatedAt)
                .organizationId(this.organizationId)
                .clientId(this.clientId)
                .componentId(this.componentId)
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
