package org.chainoptimstorage.core.warehouseinventoryitem.model;

import jakarta.persistence.*;
import lombok.*;
import org.chainoptimstorage.internal.in.goods.model.Component;
import org.chainoptimstorage.internal.in.goods.model.Product;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Entity
@Table(name = "supplier_orders")
public class WarehouseInventoryItem {


    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    @Column(name = "warehouse_id", nullable = false)
    private Integer warehouseId;

    @Column(name = "organization_id")
    private Integer organizationId;

    @ManyToOne
    @JoinColumn(name = "component_id")
    private Component component;

    @ManyToOne
    @JoinColumn(name = "product_id")
    private Product product;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    @Column(name = "quantity")
    private Float quantity;

    @Column(name = "minimum_required_quantity")
    private Float minimumRequiredQuantity;

    @Column(name = "company_id")
    private String companyId;

    public WarehouseInventoryItem deepCopy() {
        return WarehouseInventoryItem.builder()
//                .id(this.id)
                .warehouseId(this.warehouseId)
                .organizationId(this.organizationId)
                .component(this.component)
                .product(this.product)
                .createdAt(this.createdAt)
                .updatedAt(this.updatedAt)
                .quantity(this.quantity)
                .minimumRequiredQuantity(this.minimumRequiredQuantity)
                .companyId(this.companyId)
                .build();
    }
}
