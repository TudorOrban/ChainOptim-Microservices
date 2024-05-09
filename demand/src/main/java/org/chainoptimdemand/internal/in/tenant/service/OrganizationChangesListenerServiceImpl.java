package org.chainoptimdemand.internal.in.tenant.service;

import org.chainoptimdemand.shared.kafka.KafkaEvent;
import org.chainoptimdemand.shared.kafka.OrganizationEvent;
import org.chainoptimdemand.core.client.model.Client;
import org.chainoptimdemand.core.client.repository.ClientRepository;
import org.chainoptimdemand.core.clientorder.model.ClientOrder;
import org.chainoptimdemand.core.clientorder.repository.ClientOrderRepository;
import org.chainoptimdemand.core.clientshipment.model.ClientShipment;
import org.chainoptimdemand.core.clientshipment.repository.ClientShipmentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class OrganizationChangesListenerServiceImpl implements OrganizationChangesListenerService {

    private final ClientRepository clientRepository;
    private final ClientOrderRepository clientOrderRepository;
    private final ClientShipmentRepository clientShipmentRepository;

    @Autowired
    public OrganizationChangesListenerServiceImpl(ClientRepository clientRepository,
                                                  ClientOrderRepository clientOrderRepository,
                                                  ClientShipmentRepository clientShipmentRepository) {
        this.clientRepository = clientRepository;
        this.clientOrderRepository = clientOrderRepository;
        this.clientShipmentRepository = clientShipmentRepository;
    }

    @KafkaListener(topics = "${organization.topic.name:organization-events}", groupId = "demand-organization-group", containerFactory = "organizationKafkaListenerContainerFactory")
    public void listenToOrganizationEvent(OrganizationEvent event) {
        System.out.println("Organization Event in Supply: " + event);
        if (event.getEventType() != KafkaEvent.EventType.DELETE) return;

        cleanUpOrganizationEntities(event.getOldEntity().getId());
    }

    private void cleanUpOrganizationEntities(Integer organizationId) {
        List<Client> clients = clientRepository.findByOrganizationId(organizationId);
        clientRepository.deleteAll(clients);

        List<ClientOrder> clientOrders = clientOrderRepository.findByOrganizationId(organizationId);
        clientOrderRepository.deleteAll(clientOrders);

        List<Integer> clientOrderIds = clientOrders.stream()
                .map(ClientOrder::getId).toList();
        List<ClientShipment> clientShipments = clientShipmentRepository.findByClientOrderIds(clientOrderIds);
        clientShipmentRepository.deleteAll(clientShipments);
    }
}
