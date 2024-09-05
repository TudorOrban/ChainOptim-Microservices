package org.chainoptim.internalcommunication.in.storage.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class CreateWarehouseInventoryItemDTO {

    private Integer warehouseId;
    private Integer organizationId;
    private Integer productId;
    private Integer componentId;
    private String companyId;
    private Float quantity;
    private Float minimumRequiredQuantity;
}
