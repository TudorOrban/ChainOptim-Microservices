package org.chainoptimdemand.core.performance.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.chainoptimdemand.core.performance.model.ClientPerformanceReport;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class UpdateClientPerformanceDTO {

    private Integer id;
    private Integer clientId;
    private ClientPerformanceReport report;
}
