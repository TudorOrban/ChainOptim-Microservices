package org.chainoptim.core.organization.service;

import org.chainoptim.shared.kafka.OrganizationEvent;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class KafkaOrganizationServiceImpl implements KafkaOrganizationService {

    private final KafkaTemplate<String, OrganizationEvent> kafkaTemplate;

    @Autowired
    public KafkaOrganizationServiceImpl(KafkaTemplate<String, OrganizationEvent> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }

    @Value("${organization.topic.name:organization-events}")
    private String organizationTopicName;

    public void sendOrganizationEvent(OrganizationEvent organizationEvent) {
        kafkaTemplate.send(organizationTopicName, organizationEvent);
    }

    public void sendOrganizationEventsInBulk(List<OrganizationEvent> kafkaEvents) {
        kafkaEvents
                .forEach(organizationEvent -> kafkaTemplate.send(organizationTopicName, organizationEvent));
    }
}
