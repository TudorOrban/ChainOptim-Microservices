package org.chainoptimnotifications.outsidefeatures.model;

import lombok.AllArgsConstructor;
import org.chainoptimnotifications.enums.Feature;
import org.chainoptimnotifications.notification.model.KafkaEvent;

@AllArgsConstructor
public class ClientOrderEvent extends KafkaEvent<ClientOrder> {

    public ClientOrderEvent(ClientOrder newEntity, ClientOrder oldEntity, EventType eventType, Integer mainEntityId, Feature mainEntityType, String mainEntityName) {
        super(newEntity, oldEntity, Feature.CLIENT_ORDER, eventType, mainEntityId, mainEntityType, mainEntityName);
    }
}