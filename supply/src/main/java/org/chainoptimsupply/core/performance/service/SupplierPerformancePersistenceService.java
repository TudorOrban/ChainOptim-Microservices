package org.chainoptimsupply.core.performance.service;

import org.chainoptimsupply.core.performance.dto.CreateSupplierPerformanceDTO;
import org.chainoptimsupply.core.performance.dto.UpdateSupplierPerformanceDTO;
import org.chainoptimsupply.core.performance.model.SupplierPerformance;

public interface SupplierPerformancePersistenceService {

    SupplierPerformance getSupplierPerformance(Integer supplierId);
    SupplierPerformance createSupplierPerformance(CreateSupplierPerformanceDTO performanceDTO);
    SupplierPerformance updateSupplierPerformance(UpdateSupplierPerformanceDTO performanceDTO);
    void deleteSupplierPerformance(Integer id);

    SupplierPerformance refreshSupplierPerformance(Integer supplierId);
}
