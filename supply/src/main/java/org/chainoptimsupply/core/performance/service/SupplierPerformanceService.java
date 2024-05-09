package org.chainoptimsupply.core.performance.service;


import org.chainoptimsupply.core.performance.model.SupplierPerformanceReport;

public interface SupplierPerformanceService {

    SupplierPerformanceReport computeSupplierPerformanceReport(Integer supplierId);
}
