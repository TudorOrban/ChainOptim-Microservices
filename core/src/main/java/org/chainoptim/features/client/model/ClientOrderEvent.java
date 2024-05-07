package org.chainoptim.features.client.model;

import org.chainoptim.shared.enums.Feature;
import org.chainoptim.shared.kafka.KafkaEvent;
import lombok.AllArgsConstructor;

@AllArgsConstructor
public class ClientOrderEvent extends KafkaEvent<ClientOrder> {

    public ClientOrderEvent(ClientOrder newEntity, ClientOrder oldEntity, EventType eventType, Integer mainEntityId, Feature mainEntityType, String mainEntityName) {
        super(newEntity, oldEntity, Feature.CLIENT_ORDER, eventType, mainEntityId, mainEntityType, mainEntityName);
    }
}