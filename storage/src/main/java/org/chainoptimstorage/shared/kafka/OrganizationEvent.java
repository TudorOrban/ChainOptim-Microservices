package org.chainoptimstorage.shared.kafka;

import lombok.AllArgsConstructor;
import org.chainoptimstorage.shared.enums.Feature;
import org.chainoptimstorage.internal.in.tenant.model.Organization;

@AllArgsConstructor
public class OrganizationEvent extends KafkaEvent<Organization> {

    public OrganizationEvent(Organization newEntity, Organization oldEntity, EventType eventType, Integer mainEntityId, Feature mainEntityType, String mainEntityName) {
        super(newEntity, oldEntity, Feature.ORGANIZATION, eventType, mainEntityId, mainEntityType, mainEntityName);
    }
}
