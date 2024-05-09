package org.chainoptimdemand.core.performance.service;

import org.chainoptimdemand.core.performance.dto.CreateClientPerformanceDTO;
import org.chainoptimdemand.core.performance.dto.UpdateClientPerformanceDTO;
import org.chainoptimdemand.core.performance.model.ClientPerformance;

public interface ClientPerformancePersistenceService {

    ClientPerformance getClientPerformance(Integer clientId);
    ClientPerformance createClientPerformance(CreateClientPerformanceDTO performanceDTO);
    ClientPerformance updateClientPerformance(UpdateClientPerformanceDTO performanceDTO);
    void deleteClientPerformance(Integer id);

    ClientPerformance refreshClientPerformance(Integer clientId);
}
