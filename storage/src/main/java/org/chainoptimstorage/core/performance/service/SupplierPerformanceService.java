package org.chainoptimstorage.core.performance.service;


import org.chainoptimstorage.core.performance.model.SupplierPerformanceReport;

public interface SupplierPerformanceService {

    SupplierPerformanceReport computeSupplierPerformanceReport(Integer supplierId);
}
