package org.chainoptimdemand.core.performance.repository;

import org.chainoptimdemand.core.performance.model.ClientPerformance;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface ClientPerformanceRepository extends JpaRepository<ClientPerformance, Integer> {

    Optional<ClientPerformance> findByClientId(Integer clientId);
}
