package org.chainoptimsupply.internal.organization.service;

import org.chainoptimsupply.shared.kafka.OrganizationEvent;

public interface OrganizationChangesListenerService {

    void listenToOrganizationEvent(OrganizationEvent event);
}
