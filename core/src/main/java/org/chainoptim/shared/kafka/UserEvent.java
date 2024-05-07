package org.chainoptim.shared.kafka;

import org.chainoptim.core.user.model.User;
import org.chainoptim.features.supplier.model.SupplierOrder;
import org.chainoptim.shared.enums.Feature;
import lombok.AllArgsConstructor;

@AllArgsConstructor
public class UserEvent extends KafkaEvent<User> {

    public UserEvent(User newEntity, User oldEntity, EventType eventType, Integer mainEntityId, Feature mainEntityType, String mainEntityName) {
        super(newEntity, oldEntity, Feature.MEMBER, eventType, mainEntityId, mainEntityType, mainEntityName);
    }
}
