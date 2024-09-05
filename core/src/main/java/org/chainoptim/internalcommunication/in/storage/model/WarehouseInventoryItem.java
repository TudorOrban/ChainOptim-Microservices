package org.chainoptim.internalcommunication.in.storage.model;

import org.chainoptim.features.product.model.Product;
import org.chainoptim.features.productpipeline.model.Component;
import lombok.*;

import java.time.LocalDateTime;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class WarehouseInventoryItem {

    private Integer id;
    private Integer warehouseId;
    private Integer organizationId;
    private Component component;
    private Product product;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Float quantity;
    private Float minimumRequiredQuantity;
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
