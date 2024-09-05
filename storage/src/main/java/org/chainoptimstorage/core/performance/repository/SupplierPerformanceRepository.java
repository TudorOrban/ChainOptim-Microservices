package org.chainoptimstorage.core.performance.repository;

import org.chainoptimstorage.core.performance.model.SupplierPerformance;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface SupplierPerformanceRepository extends JpaRepository<SupplierPerformance, Integer> {

    Optional<SupplierPerformance> findBySupplierId(Integer supplierId);
}
