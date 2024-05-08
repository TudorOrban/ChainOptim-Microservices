package org.chainoptim.shared.kafka;

import org.chainoptim.core.organization.model.Organization;
import org.chainoptim.core.user.model.User;
import org.chainoptim.shared.enums.Feature;
import lombok.AllArgsConstructor;

@AllArgsConstructor
public class OrganizationEvent extends KafkaEvent<Organization> {

    public OrganizationEvent(Organization newEntity, Organization oldEntity, EventType eventType, Integer mainEntityId, Feature mainEntityType, String mainEntityName) {
        super(newEntity, oldEntity, Feature.ORGANIZATION, eventType, mainEntityId, mainEntityType, mainEntityName);
    }
}
