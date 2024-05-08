package org.chainoptim.core.organization.service;

import org.chainoptim.shared.kafka.OrganizationEvent;

import java.util.List;

public interface KafkaOrganizationService {

    void sendOrganizationEvent(OrganizationEvent userEvent);
    void sendOrganizationEventsInBulk(List<OrganizationEvent> kafkaEvents);
}
