package org.chainoptimnotifications.internal.in.tenant.model;

import lombok.AllArgsConstructor;
import org.chainoptimnotifications.shared.enums.Feature;
import org.chainoptimnotifications.core.notification.model.KafkaEvent;

@AllArgsConstructor
public class UserEvent extends KafkaEvent<User> {

    public UserEvent(User newEntity, User oldEntity, KafkaEvent.EventType eventType, Integer mainEntityId, Feature mainEntityType, String mainEntityName) {
        super(newEntity, oldEntity, Feature.MEMBER, eventType, mainEntityId, mainEntityType, mainEntityName);
    }
}
