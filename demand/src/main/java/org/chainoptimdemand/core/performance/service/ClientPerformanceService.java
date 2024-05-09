package org.chainoptimdemand.core.performance.service;


import org.chainoptimdemand.core.performance.model.ClientPerformanceReport;

public interface ClientPerformanceService {

    ClientPerformanceReport computeClientPerformanceReport(Integer clientId);
}
