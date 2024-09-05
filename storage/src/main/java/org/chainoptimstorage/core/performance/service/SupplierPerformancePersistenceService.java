package org.chainoptimstorage.core.performance.service;

import org.chainoptimstorage.core.performance.dto.CreateSupplierPerformanceDTO;
import org.chainoptimstorage.core.performance.dto.UpdateSupplierPerformanceDTO;
import org.chainoptimstorage.core.performance.model.SupplierPerformance;

public interface SupplierPerformancePersistenceService {

    SupplierPerformance getSupplierPerformance(Integer supplierId);
    SupplierPerformance createSupplierPerformance(CreateSupplierPerformanceDTO performanceDTO);
    SupplierPerformance updateSupplierPerformance(UpdateSupplierPerformanceDTO performanceDTO);
    void deleteSupplierPerformance(Integer id);

    SupplierPerformance refreshSupplierPerformance(Integer supplierId);
}
