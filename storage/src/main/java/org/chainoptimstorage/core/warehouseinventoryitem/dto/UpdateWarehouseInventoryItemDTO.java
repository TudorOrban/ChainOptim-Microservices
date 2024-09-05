package org.chainoptimstorage.core.warehouseinventoryitem.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class UpdateWarehouseInventoryItemDTO {

    private Integer id;
    private Integer warehouseId;
    private Integer organizationId;
    private Integer productId;
    private Integer componentId;
    private String companyId;
    private Float quantity;
    private Float minimumRequiredQuantity;
}
