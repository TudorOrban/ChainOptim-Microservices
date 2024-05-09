package org.chainoptimsupply.core.supplier.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.chainoptimsupply.core.supplierorder.model.OrderStatus;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class CreateSupplierOrderDTO {

    private Integer organizationId;
    private Integer supplierId;
    private Integer componentId;
    private Float quantity;
    private Float deliveredQuantity;
    private LocalDateTime orderDate;
    private LocalDateTime estimatedDeliveryDate;
    private LocalDateTime deliveryDate;
    private OrderStatus status;
    private String companyId;
}
