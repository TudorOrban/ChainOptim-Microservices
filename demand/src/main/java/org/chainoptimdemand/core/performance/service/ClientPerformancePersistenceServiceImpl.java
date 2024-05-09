package org.chainoptimdemand.core.performance.service;

import org.chainoptimdemand.core.performance.dto.CreateClientPerformanceDTO;
import org.chainoptimdemand.core.performance.dto.ClientPerformanceDTOMapper;
import org.chainoptimdemand.core.performance.dto.UpdateClientPerformanceDTO;
import org.chainoptimdemand.core.performance.model.ClientPerformance;
import org.chainoptimdemand.core.performance.model.ClientPerformanceReport;
import org.chainoptimdemand.core.performance.repository.ClientPerformanceRepository;
import org.chainoptimdemand.exception.ResourceNotFoundException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class ClientPerformancePersistenceServiceImpl implements ClientPerformancePersistenceService {

    private final ClientPerformanceRepository clientPerformanceRepository;
    private final ClientPerformanceService clientPerformanceService;

    @Autowired
    public ClientPerformancePersistenceServiceImpl(ClientPerformanceRepository clientPerformanceRepository,
                                                     ClientPerformanceService clientPerformanceService) {
        this.clientPerformanceRepository = clientPerformanceRepository;
        this.clientPerformanceService = clientPerformanceService;
    }

    public ClientPerformance getClientPerformance(Integer clientId) {
        return clientPerformanceRepository.findByClientId(clientId)
                .orElseThrow(() -> new ResourceNotFoundException("Client performance for client ID: " + clientId + " not found"));
    }

    public ClientPerformance refreshClientPerformance(Integer clientId) {
        // Compute fresh client performance report
        ClientPerformanceReport clientPerformanceReport = clientPerformanceService.computeClientPerformanceReport(clientId);

        ClientPerformance clientPerformance = clientPerformanceRepository.findByClientId(clientId)
                .orElse(null);

        // Create new client performance or update existing one
        if (clientPerformance == null) {
            CreateClientPerformanceDTO performanceDTO = new CreateClientPerformanceDTO();
            performanceDTO.setClientId(clientId);
            performanceDTO.setReport(clientPerformanceReport);
            return createClientPerformance(performanceDTO);
        } else {
            UpdateClientPerformanceDTO performanceDTO = new UpdateClientPerformanceDTO();
            performanceDTO.setId(clientPerformance.getId());
            performanceDTO.setClientId(clientId);
            performanceDTO.setReport(clientPerformanceReport);
            return updateClientPerformance(performanceDTO);
        }
    }
    public ClientPerformance createClientPerformance(CreateClientPerformanceDTO performanceDTO) {
        return clientPerformanceRepository.save(ClientPerformanceDTOMapper.mapCreateClientPerformanceDTOToClientPerformance(performanceDTO));
    }

    public ClientPerformance updateClientPerformance(UpdateClientPerformanceDTO performanceDTO) {
        ClientPerformance performance = clientPerformanceRepository.findById(performanceDTO.getId())
                .orElseThrow(() -> new ResourceNotFoundException("Client performance with ID: " + performanceDTO.getId() + " not found"));
        ClientPerformanceDTOMapper.setUpdateClientPerformanceDTOToClientPerformance(performanceDTO, performance);

        return clientPerformanceRepository.save(performance);
    }

    public void deleteClientPerformance(Integer id) {
        clientPerformanceRepository.deleteById(id);
    }
}
