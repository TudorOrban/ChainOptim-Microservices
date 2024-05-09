package org.chainoptimsupply.core.performance.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.chainoptimsupply.core.performance.model.SupplierPerformanceReport;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class UpdateSupplierPerformanceDTO {

    private Integer id;
    private Integer supplierId;
    private SupplierPerformanceReport report;
}
