package org.chainoptimsupply.internal.organization.service;

import org.chainoptimsupply.kafka.OrganizationEvent;

public interface OrganizationChangesListenerService {

    void listenToOrganizationEvent(OrganizationEvent event);
}
