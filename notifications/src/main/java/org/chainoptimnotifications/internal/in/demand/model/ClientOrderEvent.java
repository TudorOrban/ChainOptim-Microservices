package org.chainoptimnotifications.internal.in.demand.model;

import lombok.AllArgsConstructor;
import org.chainoptimnotifications.shared.enums.Feature;
import org.chainoptimnotifications.core.notification.model.KafkaEvent;

@AllArgsConstructor
public class ClientOrderEvent extends KafkaEvent<ClientOrder> {

    public ClientOrderEvent(ClientOrder newEntity, ClientOrder oldEntity, EventType eventType, Integer mainEntityId, Feature mainEntityType, String mainEntityName) {
        super(newEntity, oldEntity, Feature.CLIENT_ORDER, eventType, mainEntityId, mainEntityType, mainEntityName);
    }
}