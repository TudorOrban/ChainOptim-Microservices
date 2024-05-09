package org.chainoptimdemand.core.performance.dto;


import org.chainoptimdemand.core.performance.model.ClientPerformance;

public class ClientPerformanceDTOMapper {

    private ClientPerformanceDTOMapper() {}

    public static ClientPerformance mapCreateClientPerformanceDTOToClientPerformance(CreateClientPerformanceDTO dto) {
        ClientPerformance clientPerformance = new ClientPerformance();
        clientPerformance.setClientId(dto.getClientId());
        clientPerformance.setReport(dto.getReport());

        return clientPerformance;
    }

    public static void setUpdateClientPerformanceDTOToClientPerformance(UpdateClientPerformanceDTO dto, ClientPerformance clientPerformance) {
        clientPerformance.setClientId(dto.getClientId());
        clientPerformance.setReport(dto.getReport());
    }
}
