package org.chainoptimstorage.core.performance.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.chainoptimstorage.core.performance.model.SupplierPerformanceReport;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class CreateSupplierPerformanceDTO {

    private Integer supplierId;
    private SupplierPerformanceReport report;
}
