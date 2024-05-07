package org.chainoptimnotifications.outsidefeatures.model;

import lombok.AllArgsConstructor;
import org.chainoptimnotifications.enums.Feature;
import org.chainoptimnotifications.notification.model.KafkaEvent;
import org.chainoptimnotifications.user.model.User;

@AllArgsConstructor
public class UserEvent extends KafkaEvent<User> {

    public UserEvent(User newEntity, User oldEntity, KafkaEvent.EventType eventType, Integer mainEntityId, Feature mainEntityType, String mainEntityName) {
        super(newEntity, oldEntity, Feature.MEMBER, eventType, mainEntityId, mainEntityType, mainEntityName);
    }
}
