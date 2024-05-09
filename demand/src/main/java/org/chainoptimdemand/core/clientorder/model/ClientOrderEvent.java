package org.chainoptimdemand.core.clientorder.model;

import org.chainoptimdemand.shared.enums.Feature;
import org.chainoptimdemand.shared.kafka.KafkaEvent;
import lombok.AllArgsConstructor;

@AllArgsConstructor
public class ClientOrderEvent extends KafkaEvent<ClientOrder> {

    public ClientOrderEvent(ClientOrder newEntity, ClientOrder oldEntity, EventType eventType, Integer mainEntityId, Feature mainEntityType, String mainEntityName) {
        super(newEntity, oldEntity, Feature.SUPPLIER_ORDER, eventType, mainEntityId, mainEntityType, mainEntityName);
    }
}
